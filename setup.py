from setuptools import setup, find_packages

setup(
    name='richJinju',
    version='0.1.0',
    description='Setting up a python package',
    author='Dongho Jang',
    author_email='dongho18@gnu.ac.kr',
    # packages=find_packages(include=['exampleproject', 'exampleproject.*']),
    install_requires=[
        'python-dotenv==0.21.0',				# version 명시 안 함
        'requests==2.28.1',		# 정확한 version 명시
        'xmltodict==0.13.0',		# 최소 version 명시
        'pymysql==1.0.2'
    ],
    entry_points={	# my-command 라는 커맨드 명령어로 example.py의 main 함수 실행
        'console_scripts': ['my-command=exampleproject.example:main']
    },
)