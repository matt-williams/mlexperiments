import setuptools
import os
import glob
from distutils.command.build import build as _build
from setuptools.command.develop import develop as _develop
from pathlib import Path

with open("README.md", "r") as fh:
    long_description = fh.read()

class build(_build):  # pylint: disable=invalid-name
    sub_commands = _build.sub_commands + [('InstallCommands', None)]

class develop(_develop):  # pylint: disable=invalid-name
    sub_commands = _develop.sub_commands + [('InstallCommands', None)]

    def run(self):
        super(_develop, self).run()
        for cmd_name in self.get_sub_commands():
            self.run_command(cmd_name)

class InstallCommands(setuptools.Command):
    """A setuptools Command class to install complex dependencies."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def system(self, cmd):
        with os.popen(cmd) as p:
            print(p.read())
            status = p.close()
            if status:
                raise OSError(status, cmd)

    def get_platform_str(self):
        from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
        return '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

    def get_package_file(self, package):
        package_files = glob.glob('{}/{}-*-{}-*.whl'.format(Path.home(), package, get_platform_str()))
        return package_files[-1] if package_files else None

    def install_pytorch(self):
        try:
            import torch
        except ImportError:
            platform = get_platform_str()
            accelerator = 'cu80' if os.path.exists('/opt/bin/nvidia-smi') else 'cpu'
            self.system('pip install -q http://download.pytorch.org/whl/{}/torch-0.3.0.post4-{}-linux_x86_64.whl torchvision'.format(accelerator, platform))

    def install_vizdoom(self):
        try:
            import vizdoom
        except ImportError:
            self.system('apt-get update > /dev/null')
            self.system('apt-get install libboost-all-dev libsdl2-dev libbz2-dev libgtk2.0-dev > /dev/null')
            package_file = get_package_file('vizdoom')
            if package_file:
                self.system('pip install {}'.format(package_file))
            else:
                self.system('apt-get install cmake nasm > /dev/null')
                self.system('pip install vizdoom')
                self.system('cp `find ~/.cache/pip/wheels -name vizdoom*.whl` ~')

    def install_oblige(self):
        try:
            import oblige
        except ImportError:
            self.system('apt-get update > /dev/null')
            self.system('apt-get install libfltk1.3-dev libxft-dev libjpeg-dev libpng-dev zlib1g-dev > /dev/null')
            package_file = get_package_file('oblige')
            if package_file:
                self.system('pip install {}'.format(package_file))
            else:
                self.system('apt-get install g++ binutils make libxinerama-dev xdg-utils > /dev/null')
                self.system('pip install oblige')
                self.system('cp `find ~/.cache/pip/wheels -name oblige*.whl` ~')

    def run(self):
        self.install_pytorch()
        self.install_vizdoom()
        self.install_oblige()

setuptools.setup(
    name='ml_experiments',
    version='0.0.1',
    author='Matt Williams',
    author_email='matwilliams@hotmail.com',
    description='Machine learning experiments',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/matt-williams/ml_experiments',
    packages=setuptools.find_packages(),
    install_requires=[
        'pydrive',
        'scikit-image>=0.14.0'
    ],
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        'build': build,
        'develop': develop,
        'InstallCommands': InstallCommands,
    }
)
