function prod = pagemtimes_right(Pgs, M)
    prod = pagetranspose(pagemtimes_left(transpose(M), pagetranspose(Pgs)));
end
