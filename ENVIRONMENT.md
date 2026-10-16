# MaSIF environment: package versions and install recipe

This records what's installed in `venv/` so the same stack can be
reproduced in another location. Verified by installing `environment.yml`
fresh into an independent location and running the full
`compute_surface_features.py` pipeline against it (NanoShaper surface,
APBS electrostatics, MaSIF-site interface prediction) — the solver picked
different point releases than `venv/`'s own (e.g. `tensorflow=1.10.0`
instead of `1.9.0`) and it still worked end to end, which is the actual
point of using ranges instead of exact pins.

## 1. Conda/mamba environment

```bash
mamba env create --prefix /path/to/new/venv -f environment.yml
```

`environment.yml` pins only the packages MaSIF code actually
imports/invokes, each to a **range**, not an exact version — everything
else (its ~120 transitive dependencies: libgcc, openssl, zlib, etc.) is
left for the solver to pick, so it can resolve to current builds on
whatever machine you install on rather than demanding byte-identical
packages this one happens to have.

### Why the ranges are the shape they are

**`python=3.6.*` is the one hard constraint, and it caps everything else.**
`pymesh2` (mesh regularization, `.ply` I/O) has no conda-forge build for
any Python newer than 3.6 — checked directly, only `0.2.1` and `0.3` exist,
both `py36`-only. Every other package's range below is really "whatever
range is both new enough for the APIs this code uses, and old enough to
still have a Python-3.6 build" — which, because conda-forge mostly drops
old-Python builds entirely rather than keeping them in parallel, usually
collapses to a fairly narrow window near the *newest* release that still
supports 3.6. That's not an arbitrary tight pin; it's the actual last
point where "supports our Python" and "new enough" overlap. Checked
directly against the conda-forge index for each package below (not
guessed):

| Package | Range | Ceiling reason | Floor reason |
|---|---|---|---|
| `python` | `3.6.*` | n/a — this is the hard constraint | `pymesh2` has no build below 3.6 either |
| `pymesh2` | `>=0.2.1,<0.4` | Only `0.2.1`/`0.3` exist on conda-forge, both py36-only | `fix_mesh()` (`triangulation/fixmesh.py`) uses `pymesh.remove_duplicated_vertices`/`collapse_short_edges`/`compute_outer_hull` etc. — present in both builds |
| `tensorflow` | `>=1.8,<1.14` | `source/masif_modules/MaSIF_site.py` uses `tf.contrib.layers.fully_connected`/`xavier_initializer`, removed entirely in TF2 (not even in `compat.v1`) — anything TF2 is out. Conda-forge's last python-3.6 TF1 build is `1.13.1`. Verified `1.9.0` (what the README says the checkpoint was trained with) AND `1.10.0` (what the solver actually picked when re-resolving this exact range) both run the full pipeline correctly — the range is genuinely usable, not just theoretical. | `tf.Graph`/`tf.placeholder`/`tf.Session`/`tf.train.Saver` API, stable across all of TF1 |
| `absl-py` | `>=0.9,<2.0` | **Must exclude `2.0.0` itself, not just `2.1.0`+**: verified by direct testing that `tensorflow` fails to import against `absl-py=2.0.0` with `TypeError: metaclass conflict` in `absl.flags._argument_parser`, while `0.6.1`/`1.1.0`/`1.4.0` all work. (`2.1.0`+ is separately uninstallable alongside `python=3.6` at all, since conda-forge requires `python>=3.7`/`>=3.9` from `2.1.0` on — but that's not why the ceiling is `<2.0`, the `2.0.0` break is.) Conda picks `absl-py=2.0.0` by default when resolving `tensorflow` unconstrained, so this must be pinned explicitly or the install silently breaks on import. | Any absl-py new enough to be on conda-forge at all works below the ceiling |
| `networkx` | `>=2.0,<2.7.1` (i.e. up to `2.7`) | `2.7.1` is the first build on conda-forge to require `python>=3.8`; `2.7` is the newest that still supports 3.6 | `source/geometry/compute_polar_coordinates.py` uses `nx.Graph`/`nx.all_pairs_dijkstra_path_length`/`add_weighted_edges_from` — all present since networkx 2.0 |
| `scikit-learn` | `>=0.22,<0.25` | `0.24.2` is the newest conda-forge build with a `python=3.6` variant | `sklearn.manifold.MDS`, `sklearn.neighbors.KDTree`, `sklearn.metrics.roc_auc_score` — all stable since well before 0.22 |
| `biopython` | `>=1.72,<1.80` | `1.79` is the newest conda-forge build for Python 3.6 | `Bio.PDB.PDBParser`/`PDBIO`/`Selection`/`DSSP` — stable across this range |
| `numpy` | `>=1.16,<1.20` | `1.19.5` is the newest conda-forge build for Python 3.6; also don't go far below what `tensorflow`'s compiled C extensions expect | General array ops only, no version-specific API used |
| `pdb2pqr` | `>=3.1,<3.7` (**NOT 2.x**) | `source/triangulation/computeAPBS.py` targets pdb2pqr 3.x's CLI (`--ff=PARSE` uppercase, `--apbs-input <filename>` taking an explicit argument, positional `output_pqr`) — this was changed from the original pdb2pqr-2.1.1-era CLI during this work. If you only have pdb2pqr 2.1.1, those three CLI changes in `computeAPBS.py` need reverting instead. Upper bound is just "latest available", not a known break. | pdb2pqr 3.x CLI shape, introduced at 3.0 |
| `apbs` | `>=1.4` | No known upper bound; `1.5` is what conda-forge ships for linux-64 generically (not python-versioned, so it isn't capped by the Python-3.6 constraint the way the others are) | Needs the `mg-auto`/`elec` input-file keywords `computeAPBS.py` writes, present since well before 1.4 |
| `reduce` | any | No constraint found — only used for simple protonation/trim (`-Trim`, `-HIS` flags), stable CLI | ships `reduce_wwPDB_het_dict.txt`, auto-discovered by `compute_surface_features.py` |
| `ipython` | any | `IPython.core.debugger.set_trace` is used purely as a debug hook on error paths in `global_vars.py`/`protonate.py` — never actually hit unless something's already broken | — |

**`boost`/`cmake`/`gmp`/`mpfr`/`git` are deliberately NOT in `environment.yml`.**
They're only needed once, to build NanoShaper and `multivalue` from source
(section 2/3 below) — nothing in the Python code imports or calls them, and
confirmed they aren't pulled in transitively by anything that is. Including
them in the main spec made the solve pathologically slow: `cmake`'s own
transitive dependency tree (bundled curl/libarchive/openssl/zlib versions
across its many releases) combined combinatorially with the already-narrow
Python-3.6 constraint and made mamba's solver hang indefinitely (confirmed:
removing just `cmake` from an otherwise-identical spec took the solve from
"still running after 15+ minutes" to under 15 seconds). Use whatever
`cmake`/`boost`/`gmp`/`mpfr`/`git` your OS already provides (RHEL 9:
`dnf install cmake boost-devel gmp-devel mpfr-devel git`) for the one-time
build — there's no need for them to coexist with the Python environment at
all, and keeping them out keeps the Python env's own solve fast.

**If you need a wider Python version** (3.7+), the real blocker is
`pymesh2` specifically, not anything else in this list — everything else
here has newer builds for 3.7/3.8/3.9+ on conda-forge; you'd need to either
find/build a `pymesh2` wheel for your target Python, or swap `fix_mesh()`'s
mesh-regularization calls for a different library (e.g. `trimesh` +
`open3d`), which is a real code change, not just a version bump.

## 2. NanoShaper (not a conda package — built from source)

NanoShaper has no conda-forge/bioconda package. It was built directly from
its upstream GitLab repository and the resulting binary + two small runtime
libraries were copied into `venv/bin` and `venv/lib`.

- **Source**: `https://gitlab.iit.it/sdecherchi/nanoshaper`, commit
  `6fbbc68ccbb4eaadc196f1053d6e826d54ed8fe4` (branch `master`, dated
  2026-06-04). License: GPLv3 (see repo's `license.txt`) — fine to invoke as
  an external subprocess from Apache-2.0 code, but do not vendor/link its
  source into this repo.
- **Reported version**: `NanoShaper 1.5` (from its own startup banner).

### Build recipe

```bash
git clone https://gitlab.iit.it/sdecherchi/nanoshaper nanoshaper_src
cd nanoshaper_src
echo "exe" | python3 setup.py   # answers the "so/exe" prompt; builds via CMakeLists_standalone.txt
```

On this machine (RHEL 9.8, gcc 11.5.0, system `boost-{system,filesystem,thread}-1.75.0`,
system `gmp`/`mpfr` under `/usr/lib64` and `/lib64`), `setup.py`'s cmake
step found CGAL as header-only and the system Boost/GMP/MPFR automatically,
so it skipped downloading/building its own CGAL and only needed to fetch a
prebuilt CMake tarball and the oneTBB 2021.4.0 tarball (both from GitHub
releases — `wget` must be able to reach `github.com`/`codeload.github.com`).
The resulting binary is at `nanoshaper_src/build/NanoShaper`.

**System libraries required at build and run time** (not provided by conda
here — these came from the OS):
- Boost >= 1.66 with `system`/`filesystem`/`thread` components (RHEL 9:
  `dnf install boost-devel`, or just `boost-{system,filesystem,thread}` at
  runtime) — version installed: 1.75.0 (`boost-system-1.75.0-13.el9_7`).
- GMP, MPFR (RHEL 9: these are base-image libraries, `/usr/lib64/libgmp.so`,
  `/lib64/libmpfr.so`).
- A C++ compiler supporting C++14 (gcc >= 8, per NanoShaper's own README;
  gcc 11.5.0 used here).

### Required fix-up after building (do this every time you rebuild)

The build downloads oneTBB into a throwaway path
(`nanoshaper_src/oneapi-tbb-2021.4.0/...`) and the resulting `NanoShaper`
binary's rpath points there by default — **if that directory is deleted
(e.g. it's under `/tmp`), the binary stops working.** Fix:

```bash
# Copy the two TBB runtime libs NanoShaper actually needs into venv/lib
cp -P nanoshaper_src/oneapi-tbb-2021.4.0/lib/intel64/gcc4.8/libtbb.so.12.4 \
      nanoshaper_src/oneapi-tbb-2021.4.0/lib/intel64/gcc4.8/libtbb.so.12 \
      nanoshaper_src/oneapi-tbb-2021.4.0/lib/intel64/gcc4.8/libtbbmalloc.so.2.4 \
      nanoshaper_src/oneapi-tbb-2021.4.0/lib/intel64/gcc4.8/libtbbmalloc.so.2 \
      /path/to/venv/lib/

# Repoint the binary's rpath from the throwaway TBB dir to venv/lib
# (requires `patchelf`, e.g. `dnf install patchelf` or `conda install patchelf`)
patchelf --set-rpath '$ORIGIN/../lib' nanoshaper_src/build/NanoShaper

cp nanoshaper_src/build/NanoShaper /path/to/venv/bin/NanoShaper
```

After this, `ldd venv/bin/NanoShaper` should show every library resolving
from either `venv/lib` or system paths (`/lib64`, `/usr/lib64`) — nothing
under `/tmp` or any other scratch directory. Verify with:

```bash
ldd /path/to/venv/bin/NanoShaper
```

## 3. `multivalue` (not a conda package — compiled from APBS's own bundled source)

APBS ships `multivalue.c`'s source under
`venv/share/apbs/tools/mesh/multivalue.c`, but conda's `apbs` package does
not build/install a `multivalue` binary. It was compiled against the
headers/libraries the conda `apbs` package does provide:

```bash
VENV=/path/to/venv
mkdir -p /tmp/multivalue_build

# apbscfg.h ships only under tools/windows/ in this APBS source tree, but
# its content is just autoconf HAVE_* feature flags (all #undef by
# default) -- safe to reuse verbatim for a Linux build, enabling the
# standard POSIX headers:
cp "$VENV/share/apbs/tools/windows/configuration headers/apbscfg.h" /tmp/multivalue_build/
sed -i \
  -e 's/#undef HAVE_STDLIB_H/#define HAVE_STDLIB_H 1/' \
  -e 's/#undef HAVE_STRING_H/#define HAVE_STRING_H 1/' \
  -e 's/#undef HAVE_UNISTD_H/#define HAVE_UNISTD_H 1/' \
  -e 's/#undef HAVE_SYS_TYPES_H/#define HAVE_SYS_TYPES_H 1/' \
  -e 's/#undef HAVE_SYS_STAT_H/#define HAVE_SYS_STAT_H 1/' \
  -e 's/#undef HAVE_TIME_FUNC/#define HAVE_TIME_FUNC 1/' \
  -e 's/#undef HAVE_RAND_FUNC/#define HAVE_RAND_FUNC 1/' \
  -e 's/#undef HAVE_SRAND_FUNC/#define HAVE_SRAND_FUNC 1/' \
  /tmp/multivalue_build/apbscfg.h

gcc -O2 -I/tmp/multivalue_build -I$VENV/include -I$VENV/include/apbs \
    $VENV/share/apbs/tools/mesh/multivalue.c \
    -L$VENV/lib -lapbs_generic -lapbs_mg -lapbs_routines -lapbs_pmgc -lmaloc -lm \
    -Wl,-rpath,$VENV/lib \
    -o $VENV/bin/multivalue
chmod +x $VENV/bin/multivalue
```

This only needs redoing if the `apbs` conda package version changes (its
headers/libs are what `multivalue.c` links against).

## 4. The pretrained MaSIF-site checkpoint

Already committed in this repo at
`data/masif_site/nn_models/all_feat_3l/model_data/model.{index,meta,data-00000-of-00001}`
— no separate installation needed; `compute_surface_features.py`
auto-discovers it relative to the repo root. If this repo is copied
elsewhere, that directory just needs to come along with it (it's checked
into git, not `.gitignore`d).

## 5. Verifying a new install

```bash
VENV=/path/to/new/venv
export PATH=$VENV/bin:$PATH
$VENV/bin/python -W ignore source/data_preparation/compute_surface_features.py \
    some_test.pdb /tmp/test_out
```

With zero extra flags/env vars (when run with the venv's own Python and
`NanoShaper`/`multivalue` sitting in `venv/bin`), this should produce a
`.ply` with populated `charge`, `hbond`, `hphob`, `iface`, and `nx/ny/nz`
columns, and exit 0. See `source/data_preparation/compute_surface_features.py --help`
for all options.
