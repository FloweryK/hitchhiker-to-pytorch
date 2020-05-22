# The Hitchhiker's Guide to the Pytorch (Korean)

The Hitchhiker's Guide to the Pytorch

토치를 여행하는 머신러닝 뉴비들을 위한 불친절한 안내서.

사실 뉴비인 저를 위한 안내서입니다.



## Table of Contents

1. [CNN으로 cifar10 데이터 구분하기](01-cnn-cifar10-classifier/README.md)
2. [GAN으로 MNIST 데이터 만들기](02-gan-mnist-generator/README.md)
3. 모델 저장하기, 로드하기
4. 자신의 데이터를 사용하기 (custom Dataset, Dataloader)
5. [CUDA를 이용해 GPU로 훈련하기 (단일 처리, 병렬 처리)](cuda-gpu-parallelism/README.md)



## Before we start

* 이 레포지토리의 코드들은 모두 python3, torch를 사용합니다. python3.6.8 이상을 사용하시는 걸 추천합니다. 
* 각 디렉토리는 데이터셋을 포함하고 있지 않습니다. 코드를 실행시키면 맨 처음에만 예제 데이터를 다운로드하게 됩니다!



## Installing

1. 우선 다음을 통해 프로젝트를 가져오도록 합니다.

   ```bash
   $ git clone https://github.com/FloweryK/hitchhiker-to-pytorch.git
   ```

2. (옵션: 가상환경 사용하기) 
   다음을 통해 가상환경을 만들어 프로젝트 내에서만 사용하는 라이브러리들을 관리할 수 있습니다.

   ```bash
   $ virtualenv venv -p python3
   $ source venv/bin/activate # 가상환경 켜기 (linux, mac)
   $ source venv/Scripts/activate # 가상환경 켜기 (Windows)
   $ deactivate # 가상환경 끄기
   ```

3. 다음을 통해 필요한 라이브러리들을 설치합니다. 

   ```bash
   $ pip3 install -r requirements.txt
   ```

   

## How to run the codes

코드의 설명과 작동 방식은 각 디렉토리 내 README.md에서 설명합니다. 



## Troubleshooting

##### [Windows] import torch 에러 (DLL load failed, WinError 126)

```bash
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  ...
  ...
    self._handle = _dlopen(self._name, mode)
OSError: [WinError 126] The specified module could not be found
```

윈도우 업데이트가 제대로 되지 않은 경우 발생하는 에러입니다. Process Monitor로 확인하면, MSVCP140.dll을 찾을 수 없어 로드하지 못하는 것을 알 수 있습니다. 

마이크로소프트 공식 홈페이지 [Visual Studio 2015용 Visual C++ 재배포 가능 패키지](https://www.microsoft.com/ko-kr/download/details.aspx?id=48145)에서 파일을 다운받아 설치하시면 정상적으로 로드 할 수 있습니다. 



