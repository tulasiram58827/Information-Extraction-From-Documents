from setuptools import setup

setup(
    install_requires=[
        'pandas',
        'google-cloud-vision',
        'matplotlib',
        'tensorflow',
        'tensorflow-addons',
        'opencv-python',
        'imutils',
        'tqdm',
        'dateparser',
        'wandb'
    ],
    python_requires='>=3.8',
)
