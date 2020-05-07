# CUDA GPU usage, Data Parallelism


## How to run the code

```bash
$ python3 main.py
```



## Explanations

1. (1)

   ```python
   # Configurations
   N_BATCH = 100 * torch.cuda.device_count() if torch.cuda.device_count() else 100
   ```

2. (2)

   ```python
   # define networks
   G = Generator()
   D = Discriminator()
   
   # if cuda and gpu is available, use them. Otherwise, use CPU as usual.
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   G.to(device)
   D.to(device)
   ```

3. (3)

   ```
   if torch.cuda.device_count() > 1:
       print('currently using ' + str(torch.cuda.device_count()) + ' cuda devices.')
       G = nn.DataParallel(G)
       D = nn.DataParallel(D)
   ```

   







## References

