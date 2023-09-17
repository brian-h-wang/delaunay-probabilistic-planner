from distutils.core import setup

setup(
    name="mhp-planner",
    version="0.1",
    description="Multiple-hypothesis planner.",
    url="https://github.com/brian-h-wang/multiple-hypothesis-planner/",
    author="Brian H. Wang",
    author_email="bhw45@cornell.edu",
    license="MIT",
    packages=["planner", "experiments"],
    install_requires=['gtsam']
)

