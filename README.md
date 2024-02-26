## 1. Setup 

### 1.1 Apple M1

#### 1.1.2 Dependant Libraries

1. brew install open-mpi
2. pip install --no-cache-dir mpi4py
3. pip install cmake
4. pip install swig
5. brew install freetype
6. Command Line Tools for xcode: xcode-select --install (The dialog box says it'll take 200+ hours to complete so install it externally - https://developer.apple.com/download/all/?q=command)


#### 1.1.3 Install OpenAI's Spining Up
```
pip install git+https://github.com/nirajpandkar/spinningup.git@2023Jan_dependency_upgrades
``````
