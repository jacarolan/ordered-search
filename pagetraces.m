function traces = pagetraces(Pgs)
    pg_sz = size(Pgs, 1);
    pg_cnt = size(Pgs, 3);
    
    diag_idx = (1:pg_sz+1:pg_sz^2).' + (0:pg_cnt-1)*pg_sz^2;
    traces = sum(Pgs(diag_idx), 1);
end