from distutils.core import setup


if __name__ == '__main__':
    setup(
        name='hrs',
        version='1.1.0',
        author='Bo Zhang',
        author_email='bozhang@nao.cas.cn',
        # py_modules=['hrs'],
        description='High Resolution Spectrograph (2.16m) Reduction pipeline.',  # short description
        license='New BSD',
        # install_requires=['numpy>=1.7','scipy','matplotlib','nose'],
        url='http://github.com/hypergravity/hrs',
        classifiers=[
            "Development Status :: 6 - Mature",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 2.7",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Topic :: Scientific/Engineering :: Physics"],
        package_dir={'hrs/': ''},
        packages=['hrs', 'song'],
        # package_data={'hrs/data': [''],
        #               "":          ["LICENSE"]},
        include_package_data=True,
        requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'skimage',
                  'joblib', 'ccdproc', 'tqdm']
    )
