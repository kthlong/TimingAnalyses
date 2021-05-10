% classify_dir.m

% sel_amp =1  ;


%SEL AMP is force
%DIRindex is Texture

load('Rstimes_80')
num_dir = 4;      %length(dirs);
dir_index=[12 2 30 37];
%temp_res = [0.0005 0.001 0.0025 0.005 0.0075 0.01 0.05 0.1 0.5 1];
%q = 1./temp_res;
temp_res= [.2;0.0910201934670620;0.0308438473287232;0.0104519984170529;0.00354184968385392;0.00120022015718533];
q= [0 10.9866   32.4214   95.6755  282.3384  833.1805];
%q(end) = 0;
begtime=2;
endtime=2.5;

z = 1;
N=7;
tic 
CM = zeros(num_dir,num_dir,N,length(q));
for qq=1:length(q)
    for i=1:N  
        for d1=1:num_dir
            for r1=1:length(Rstimes{i}{sel_amp,dir_index(d1)})
                st_cur = Rstimes{i}{sel_amp,dir_index(d1)}{r1};    %change sel amp or dir
                st_cur = st_cur(st_cur>begtime & st_cur<endtime);
                
                for d2=1:num_dir
                    c = 1;
                    costs = [];
                    for r2=1:length(Rstimes{i}{sel_amp,dir_index(d2)})
                        if d1==d2 && r1==r2
                            continue;
                        end
                        st_cmp = Rstimes{i}{sel_amp+together,dir_index(d2)}{r2};
                        st_cmp = st_cmp(st_cmp>begtime & st_cmp<endtime);
                        
                        
                        %d_tmp = spkdl([st_cur'; st_cmp'],[1 length(st_cur)+1],[length(st_cur) length(st_cur)+length(st_cmp)],q(qq));
                        %costs(c) = d_tmp(2);
                        costs(c) = spkd_slide(st_cur',st_cmp',q(qq),0.05,0.001);
                        
                        c = c+1;
                    end
                    mCosts(d2) = mean(costs.^z).^(1/z);
                end
                [~,mind] = min(mCosts);
                CM(d1,mind,i,qq) = CM(d1,mind,i,qq)+1;
            end
            CM(d1,:,i,qq) = CM(d1,:,i,qq)/length(Rstimes{i}{sel_amp,d1});
        end
        acc(i,qq) = mean(diag(CM(:,:,i,qq)));
    end
    fprintf('.')
end
fprintf('\n')
toc

%% save results


