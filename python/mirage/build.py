import contextlib
import functools
import io
import os
import shutil
import subprocess
import sys
import sysconfig

# code from Triton

@functools.lru_cache()
def libcuda_dirs():
    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so"))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the file.'
    else:
        msg += 'Please make sure GPU is setup and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so')) for path in dirs), msg
    return dirs

@functools.lru_cache()
def cuda_include_dir():
    base_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
    cuda_path = os.path.join(base_dir, "third_party", "cuda")
    return os.path.join(cuda_path, "include")


def build(name, src, srcdir):
    cuda_lib_dirs = libcuda_dirs()
    cu_include_dir = cuda_include_dir()
    
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    # TODO: support more things here.
    nvcc = shutil.which("nvcc")
    cc = nvcc
    if cc is None:
        raise RuntimeError("Failed to find nvcc.")
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]

    cc_cmd = [
        cc, src, "-O3", f"-I{py_include_dir}", f"-I{srcdir}", "-shared", "-fPIC", "-lcuda",
        "-o", so
    ]
    cc_cmd += [f"-L{dir}" for dir in cuda_lib_dirs]
    print('Command:', ' '.join(cc_cmd))
    return None
    ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so
    else:
        raise RuntimeError("Failed to build %s" % src)

if __name__ == '__main__':
    build("cuda_utils", "test_excepted.cu", ".")
    # print("CUDA utils built successfully")