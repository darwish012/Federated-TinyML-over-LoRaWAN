"""provision_board.py: the PC side of PROTOCOL_V4 6.1, against a fake serial port (--dry-run too)."""
from __future__ import annotations

import json
import re
import sys
import types
import zlib

import pytest

import provision_board as pb
from keys_v4 import derive_app_key, provisioning_crc, KEYS_CRC_BYTES
from roster_v4 import Roster

# Test-only values (not real keys).
MASTER = bytes(range(16))
MC_ADDR = "01020304"
MC_NWK = "0102030405060708090a0b0c0d0e0f10"
MC_APP = "a1a2a3a4a5a6a7a8a9aaabacadaeafb0"
JOIN_EUI = "70b3d57ed0010203"
DEV = "70B3D57ED0000042"


class Clock:
    """A clock that advances 1 s per reading, so a silent board times out without real waiting."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1.0
        return self.t


class Recording:
    """Wraps a fake board and records exactly what the PC wrote."""

    def __init__(self, board):
        self.board = board
        self.written: list[str] = []

    def readline(self):
        return self.board.readline()

    def write(self, data):
        self.written.append(data.decode("ascii"))
        return self.board.write(data)

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture
def material():
    return pb.KeyMaterial(joineui=bytes.fromhex(JOIN_EUI), mc_devaddr=bytes.fromhex(MC_ADDR),
                          mc_nwkskey=bytes.fromhex(MC_NWK), mc_appskey=bytes.fromhex(MC_APP))


@pytest.fixture
def roster(tmp_path):
    return Roster(tmp_path / "roster.json")


def run(board, material, roster, **kwargs):
    lines = []
    port = Recording(board)
    result = pb.provision(port, material, MASTER, roster, out=lines.append, clock=Clock(),
                          hello_timeout=kwargs.pop("hello_timeout", 30), reply_timeout=kwargs.pop("reply_timeout", 10),
                          **kwargs)
    return result, port, lines


# ---------------------------------------------------------------------------
# The exchange
# ---------------------------------------------------------------------------
def test_the_happy_path_derives_the_key_sends_keys_verifies_the_crc_and_appends_to_the_roster(material, roster):
    board = pb.FakeBoardSerial(DEV, fw_version="7")
    result, port, lines = run(board, material, roster)
    assert result == pb.Provisioned(DEV, "7", True)
    assert roster.contains(DEV) and json.loads(roster.path.read_text())["devices"][0]["dev_eui"] == DEV
    # what the board stored is what the contract says: AppKey = AES-128-ECB(master, DevEUI || 0x00 x 8)
    assert board.stored["appkey"] == derive_app_key(MASTER, DEV)
    assert board.stored == {"joineui": bytes.fromhex(JOIN_EUI), "appkey": derive_app_key(MASTER, DEV),
                            "mcaddr": bytes.fromhex(MC_ADDR), "mcnwk": bytes.fromhex(MC_NWK),
                            "mcapp": bytes.fromhex(MC_APP)}
    assert len(port.written) == 1


def test_the_keys_line_has_the_exact_wire_format(material, roster):
    board = pb.FakeBoardSerial(DEV)
    _, port, _ = run(board, material, roster)
    line = port.written[0]
    assert line.endswith("\n") and line.count("\n") == 1
    parts = line.strip().split(" ")
    assert parts[:2] == ["TFPROV", "KEYS"] and [len(p) for p in parts[2:]] == [16, 32, 8, 32, 32]
    assert all(re.fullmatch(r"[0-9a-f]+", p) for p in parts[2:])
    assert parts[2:] == [JOIN_EUI, derive_app_key(MASTER, DEV).hex(), MC_ADDR, MC_NWK, MC_APP]


def test_the_crc_covers_the_raw_values_in_keys_order():
    appkey = derive_app_key(MASTER, DEV)
    raw = bytes.fromhex(JOIN_EUI) + appkey + bytes.fromhex(MC_ADDR) + bytes.fromhex(MC_NWK) + bytes.fromhex(MC_APP)
    assert len(raw) == KEYS_CRC_BYTES == 60
    assert provisioning_crc(bytes.fromhex(JOIN_EUI), appkey, bytes.fromhex(MC_ADDR), bytes.fromhex(MC_NWK),
                            bytes.fromhex(MC_APP)) == zlib.crc32(raw) & 0xFFFFFFFF
    assert provisioning_crc(bytes.fromhex(JOIN_EUI), appkey, bytes.fromhex(MC_ADDR), bytes.fromhex(MC_APP),
                            bytes.fromhex(MC_NWK)) != zlib.crc32(raw)         # order matters


def test_boot_noise_lowercase_deveui_and_repeated_hello_lines_are_tolerated(material, roster):
    board = pb.FakeBoardSerial(DEV.lower(), noise_lines=6)
    result, _, _ = run(board, material, roster)
    assert result.dev_eui == DEV


def test_provisioning_twice_keeps_one_roster_entry(material, roster):
    first, _, _ = run(pb.FakeBoardSerial(DEV), material, roster)
    second, _, lines = run(pb.FakeBoardSerial(DEV), material, roster)
    assert first.new_in_roster is True and second.new_in_roster is False
    assert len(roster) == 1 and any("already listed" in line for line in lines)
    run(pb.FakeBoardSerial("70B3D57ED0000043"), material, roster)
    assert roster.dev_euis() == [DEV, "70B3D57ED0000043"]


def test_a_crc_mismatch_aborts_and_leaves_the_roster_alone(material, roster):
    with pytest.raises(pb.ProvisionError, match="CRC mismatch"):
        run(pb.FakeBoardSerial(DEV, fault="badcrc"), material, roster)
    assert not roster.contains(DEV) and not roster.path.exists()


def test_a_board_error_aborts_and_its_text_is_scrubbed_of_keys(material, roster):
    with pytest.raises(pb.ProvisionError, match="nvs_write_failed"):
        run(pb.FakeBoardSerial(DEV, fault="err"), material, roster)
    with pytest.raises(pb.ProvisionError) as err:
        run(pb.FakeBoardSerial(DEV, fault="err_leaks"), material, roster)
    assert derive_app_key(MASTER, DEV).hex() not in str(err.value) and "<redacted>" in str(err.value)
    assert not roster.contains(DEV)


def test_timeouts_without_hello_or_confirmation(material, roster):
    with pytest.raises(pb.ProvisionError, match="no TFPROV HELLO"):
        run(pb.FakeBoardSerial(DEV, fault="silent"), material, roster, hello_timeout=5)
    with pytest.raises(pb.ProvisionError, match="no TFPROV OK"):
        run(pb.FakeBoardSerial(DEV, fault="no_ok"), material, roster, reply_timeout=5)
    assert not roster.path.exists()


def test_expected_deveui_must_match(material, roster):
    with pytest.raises(pb.ProvisionError, match="not the expected"):
        run(pb.FakeBoardSerial(DEV), material, roster, expected_dev_eui="70B3D57ED0000099")
    assert run(pb.FakeBoardSerial(DEV), material, roster, expected_dev_eui=DEV.lower())[0].dev_eui == DEV


def test_a_malformed_hello_is_not_trusted(material, roster):
    class Garbage:
        def __init__(self):
            self.n = 0

        def readline(self):
            self.n += 1
            return b"TFPROV HELLO 1 70B3D57E\n" if self.n < 50 else b""

        def write(self, data):
            raise AssertionError("keys must not be sent to a board with a malformed DevEUI")

    with pytest.raises(pb.ProvisionError, match="malformed DevEUI"):
        pb.provision(Garbage(), material, MASTER, roster, clock=Clock(), out=lambda s: None)


def test_nothing_the_tool_prints_contains_a_key(material, roster):
    secrets = {MASTER.hex(), derive_app_key(MASTER, DEV).hex(), MC_NWK, MC_APP}
    _, port, lines = run(pb.FakeBoardSerial(DEV), material, roster)
    text = "\n".join(lines).lower()
    for secret in secrets:
        assert secret.lower() not in text
    assert MC_ADDR not in text
    # the board's confirmation line is not echoed either (it carries a CRC of the keys)
    assert "tfprov ok" not in text and "keys sent (not shown)" in text


# ---------------------------------------------------------------------------
# Environment and key material
# ---------------------------------------------------------------------------
def test_key_material_comes_from_the_environment_and_errors_never_echo_values():
    env = {pb.ENV_MC_DEVADDR: MC_ADDR, pb.ENV_MC_NWKSKEY: MC_NWK, pb.ENV_MC_APPSKEY: MC_APP}
    m = pb.KeyMaterial.from_env(JOIN_EUI, env)
    assert (m.joineui, m.mc_devaddr, m.mc_nwkskey, m.mc_appskey) == (
        bytes.fromhex(JOIN_EUI), bytes.fromhex(MC_ADDR), bytes.fromhex(MC_NWK), bytes.fromhex(MC_APP))
    for key, bad in ((pb.ENV_MC_DEVADDR, "26011b"), (pb.ENV_MC_NWKSKEY, "zz" * 16), (pb.ENV_MC_APPSKEY, "")):
        with pytest.raises(pb.ProvisionError) as err:
            pb.KeyMaterial.from_env(JOIN_EUI, dict(env, **{key: bad}))
        assert key in str(err.value) and (bad not in str(err.value) or bad == "")
    with pytest.raises(pb.ProvisionError, match="JoinEUI"):
        pb.KeyMaterial.from_env("1234", env)


# ---------------------------------------------------------------------------
# The command line, including --dry-run
# ---------------------------------------------------------------------------
@pytest.fixture
def cli(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text(f'[enrollment]\njoin_eui = "{JOIN_EUI.upper()}"\n[v4]\nroster_path = "cfg_roster.json"\n')
    for name, value in (("TINYFL_MASTER_KEY", MASTER.hex()), (pb.ENV_MC_DEVADDR, MC_ADDR),
                        (pb.ENV_MC_NWKSKEY, MC_NWK), (pb.ENV_MC_APPSKEY, MC_APP)):
        monkeypatch.setenv(name, value)
    return config


def test_dry_run_exercises_the_whole_exchange_without_touching_the_roster(cli, tmp_path, capsys):
    code = pb.main(["--dry-run", "--config", str(cli)])
    out = capsys.readouterr()
    assert code == 0 and "DRY RUN" in out.out and f"provisioned {DEV}" in out.out
    assert "roster not written" in out.out and not (tmp_path / "cfg_roster.json").exists()
    for secret in (MASTER.hex(), derive_app_key(MASTER, DEV).hex(), MC_NWK, MC_APP, MC_ADDR):
        assert secret not in out.out + out.err


def test_dry_run_with_an_explicit_roster_appends(cli, tmp_path, capsys):
    target = tmp_path / "explicit.json"
    assert pb.main(["--dry-run", "--config", str(cli), "--roster", str(target), "--fake-deveui", "70B3D57ED00000AA"]) == 0
    assert json.loads(target.read_text())["devices"][0]["dev_eui"] == "70B3D57ED00000AA"
    assert "roster updated" in capsys.readouterr().out


@pytest.mark.parametrize("fault", ["badcrc", "err", "err_leaks"])
def test_dry_run_faults_fail_with_exit_1_and_no_roster_entry(cli, tmp_path, capsys, fault):
    target = tmp_path / "r.json"
    code = pb.main(["--dry-run", "--config", str(cli), "--roster", str(target), "--fake-fault", fault])
    err = capsys.readouterr().err
    assert code == 1 and "provisioning FAILED" in err and not target.exists()
    assert derive_app_key(MASTER, DEV).hex() not in err


def test_missing_or_malformed_environment_is_exit_2(cli, monkeypatch, capsys):
    monkeypatch.delenv("TINYFL_MASTER_KEY")
    assert pb.main(["--dry-run", "--config", str(cli)]) == 2
    assert "TINYFL_MASTER_KEY" in capsys.readouterr().err
    monkeypatch.setenv("TINYFL_MASTER_KEY", "abc")
    assert pb.main(["--dry-run", "--config", str(cli)]) == 2
    monkeypatch.setenv("TINYFL_MASTER_KEY", MASTER.hex())
    monkeypatch.delenv(pb.ENV_MC_NWKSKEY)
    assert pb.main(["--dry-run", "--config", str(cli)]) == 2
    err = capsys.readouterr().err
    assert pb.ENV_MC_NWKSKEY in err and MASTER.hex() not in err


def test_missing_config_and_missing_port(cli, tmp_path, capsys):
    assert pb.main(["--dry-run", "--config", str(tmp_path / "nope.toml")]) == 2
    assert pb.main(["--config", str(cli)]) == 2 and "--port is required" in capsys.readouterr().err


def test_a_real_port_goes_through_pyserial_and_writes_the_config_roster(cli, tmp_path, monkeypatch, capsys):
    opened = []

    class SerialException(Exception):
        pass

    def Serial(port, baud, timeout):
        opened.append((port, baud, timeout))
        return pb.FakeBoardSerial("70B3D57ED00000BB")

    fake = types.SimpleNamespace(Serial=Serial, SerialException=SerialException)
    monkeypatch.setitem(sys.modules, "serial", fake)
    assert pb.main(["--port", "COM7", "--config", str(cli)]) == 0
    assert opened == [("COM7", 115200, 1)]
    assert Roster(tmp_path / "cfg_roster.json").contains("70B3D57ED00000BB")

    def broken(port, baud, timeout):
        raise SerialException("access denied")

    fake.Serial = broken
    assert pb.main(["--port", "COM7", "--config", str(cli)]) == 2
    assert "cannot open COM7" in capsys.readouterr().err


def test_pyserial_is_declared_in_the_requirements():
    text = (pb.Path(__file__).resolve().parents[1] / "requirements.txt").read_text().lower()
    assert "pyserial" in text
