def hanoi(n_disks, a=1, b=2, c=3):
    assert n_disks > 0, "n_disks at least 1"
    if n_disks == 1:
        print("move disk from %d to %d" % (a, b))
    else:
        hanoi(n_disks - 1, a, c, b)
        print("move disk from %d to %d" % (a, b))
        hanoi(n_disks - 1, c, b, a)

