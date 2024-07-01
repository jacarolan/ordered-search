function prod = pagemtimes_left(M, Pgs)
    pg_sz = size(Pgs, 1);
    pg_cnt = size(Pgs, 3);
    prod = reshape(M * reshape(Pgs, pg_sz, []), pg_sz, pg_sz, pg_cnt);
end