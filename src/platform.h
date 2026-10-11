/* Small OS helpers for the command-line tool. All paths are UTF-8. */
#ifndef IE_PLATFORM_H
#define IE_PLATFORM_H

#include <stddef.h>

double pf_now(void); /* monotonic seconds */
int pf_exe_dir(char *buf, size_t n);
int pf_home_dir(char *buf, size_t n);
const char *pf_getenv(const char *name); /* NULL when unset or empty */
int pf_exists(const char *path);
int pf_is_dir(const char *path);
int pf_mkdirs(const char *path);
int pf_rename(const char *from, const char *to); /* replaces `to` */
int pf_remove(const char *path);

/* Regular files in dir (names only, sorted). Free with pf_free_list. */
int pf_list_dir(const char *dir, char ***names, int *count);
void pf_free_list(char **names, int count);

/* Run argv[0] (searched on PATH) and wait. Returns the exit code, or -1 if it
 * could not be started. */
int pf_run(const char *const *argv);

/* Windows: re-read the command line as UTF-8 and switch the console to UTF-8.
 * Elsewhere this does nothing. */
void pf_init_args(int *argc, char ***argv);

#ifdef _WIN32
#define PF_SEP '\\'
#else
#define PF_SEP '/'
#endif

#endif
