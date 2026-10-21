// All implementation is in a header to avoid Arduino's automatic prototype
// generation interfering with portable C++ types. COMPILE ONLY in this task.
#include "node_v4.h"
#ifdef SET_LOOP_TASK_STACK_SIZE
// Training, replay and checkpoint code need more than the default 8 KB loopTask stack.
SET_LOOP_TASK_STACK_SIZE(24 * 1024);
#endif
