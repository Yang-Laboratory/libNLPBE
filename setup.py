from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from setuptools.extension import Extension
import libnlpbe
import os
import sys

# Copied from PySCF
def get_platform():
    from distutils.util import get_platform
    platform = get_platform()
    if sys.platform == 'darwin':
        arch = os.getenv('CMAKE_OSX_ARCHITECTURES')
        if arch:
            osname = platform.rsplit('-', 1)[0]
            if ';' in arch:
                platform = f'{osname}-universal2'
            else:
                platform = f'{osname}-{arch}'
        elif os.getenv('_PYTHON_HOST_PLATFORM'):
            # the cibuildwheel environment
            platform = os.getenv('_PYTHON_HOST_PLATFORM')
            if platform.endswith('arm64'):
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'arm64')
            elif platform.endswith('x86_64'):
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'x86_64')
            else:
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'arm64;x86_64')
    return platform

# Copied from PySCF
class CMakeBuildPy(build_py):
    def run(self):
        self.plat_name = get_platform()
        self.build_base = 'build'
        self.build_lib = os.path.join(self.build_base, 'lib')
        self.build_temp = os.path.join(self.build_base, f'temp.{self.plat_name}')

        self.announce('Configuring extensions', level=3)
        src_dir = os.path.abspath(os.path.join(__file__, '..', 'libnlpbe', 'lib'))
        cmd = ['cmake', f'-S{src_dir}', f'-B{self.build_temp}']
        configure_args = os.getenv('CMAKE_CONFIGURE_ARGS')
        if configure_args:
            cmd.extend(configure_args.split(' '))
        self.spawn(cmd)

        self.announce('Building binaries', level=3)
        # By default do not use high level parallel compilation.
        # OOM may be triggered when compiling certain functionals in libxc.
        # Set the shell variable CMAKE_BUILD_PARALLEL_LEVEL=n to enable
        # parallel compilation.
        cmd = ['cmake', '--build', self.build_temp]
        build_args = os.getenv('CMAKE_BUILD_ARGS')
        if build_args:
            cmd.extend(build_args.split(' '))
        if self.dry_run:
            self.announce(' '.join(cmd))
        else:
            self.spawn(cmd)
        super().run()

setup(
    name='libnlpbe',
#    version=fcdft.__version__,
    packages=find_packages(),
    include_package_data=True,
    cmdclass={'build_py': CMakeBuildPy},
)

