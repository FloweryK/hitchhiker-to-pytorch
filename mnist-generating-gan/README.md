## MNIST 데이터셋

![Mnist Examples][img/MnistExamples.png]



## How to run the code

```bash
$ python3 main.py
```

1. 맨 처음 코드를 작동시킬 경우, 코드는 data 디렉토리를 만들어 MNIST 데이터셋을 다운로드합니다. 

2. 다운로드가 끝나면, 바로 학습이 시작됩니다. 세부 설정은 main.py 내 Configurations에서 확인할 수 있습니다. 초기 설정은 다음과 같습니다:

   ```python
   # Configurations
   N_BATCH = 100			# batch 개수
   N_CLASSES = 10		# class 개수 (0부터 9까지니까 총 10개)
   D_INPUT = 28*28		# discriminator에 넣을 input 사이즈 (이미지 사이즈)
   D_NOISE = 128			# generator에 넣을 noisy input 사이즈
   N_EPOCH = 200			# 학습 회수
   lr = 0.001				# 학습 속도
   ```

3. 매 epoch마다 generator가 만드는 숫자 샘플들을 samples 디렉토리에서 확인할 수 있습니다. 



## Results (samples)



