def build_simple_conv(nfeature=4,kernel_size=3,read_size=256):

    final_size = read_size - 2 *(kernel_size-1)
    #print(final_size *2* nfeature)
    seq_modules = nn.Sequential(
        nn.Conv1d(1,nfeature,kernel_size=kernel_size),
        nn.Conv1d(nfeature,nfeature*2,kernel_size=kernel_size),
        nn.Flatten(),
        nn.Linear(final_size *2* nfeature,3))
    return seq_modules
