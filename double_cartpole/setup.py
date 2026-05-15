from setuptools import setup, find_packages
import os

DESCRIPTION = 'Gymnasium environment designed for training RL agents to balance double CartPole.'
LONG_DESCRIPTION = ('This package contains a Gymnasium environment designed for training RL agents to balance double CartPole. '
                    'The environment is automatically registered under id: double-cartpole-custom-v0, '
                    'so it can be easily used by RL agent training libraries, such as StableBaselines3.<br /><br />At the '
                    'https://github.com/marek-robak/Double-cartpole-custom-gym-env-for-reinforcement-learning.git you can find a '
                    'detailed description of the environment, along with a description of the package installation and sample '
                    'code made to train and evaluate agents in this environment.<br /><br />This environment was created for '
                    'the needs of my bachelor\'s thesis, available at https://www.ap.uj.edu.pl/diplomas/151837/ site.')

setup(
    name='double_cartpole_custom_gym_env',
    version='1.1.3',
    author='Marek Robak',
    author_email='maro.robak@gmail.com',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/marek-robak/Double-cartpole-custom-gym-env-for-reinforcement-learning.git',
    download_url='https://pypi.org/project/double-cartpole-custom-gym-env/',
    packages=find_packages(),
    include_package_data = True,
    install_requires=['gymnasium>=1.1,<1.3', 'pygame>=2.5', 'pymunk>=6.8', 'numpy>=1.26,<2.0'],
    python_requires='>=3.10',
    keywords=['reinforcement learning', 'gymnasium environment', 'StableBaselines3']
)
