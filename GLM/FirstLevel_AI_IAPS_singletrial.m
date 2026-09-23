clear

datapath = 'F:\yujun\projects\AI_IAPS\GLM\betas';
subjects = {'Sub1','Sub2', 'Sub3'};
nsub = length(subjects);

for s=1:1
    
    subj_dir = fullfile(datapath,subjects{s});
    nsess = 10;
    
    cd(subj_dir);
    cwd=pwd;
    mov=[0 0 0 0 0 0];
    mns=[0 0 0 0 0 0 0 0 0 0];

    %main effect
    cname{1} = 'Pl-AI_Pl';
    simp = [ 1 0 0 -1 0 0];
    cons{1} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype{1} = 'T';
    
    cname{2} = 'Nt-AI_Nt';
    simp = [ 0 1 0 0 -1 0];
    cons{2} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype{2} = 'T';
  
    cname{3} = 'Up-AI_Up';
    simp = [ 0 0 1 0 0 -1];
    cons{3} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype{3} = 'T';
    
    cname{4} = 'Pl-Nt';
    simp = [ 1 -1 0 0 0 0];
    cons{4} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype{4} = 'T';
    
    cname{5} = 'Pl-Up';
    simp = [ 1 0 -1 0 0 0];
    cons{5} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype{5} = 'T';
  
    cname{6} = 'Up-Nt';
    simp = [ 0 -1 1 0 0 0];
    cons{6} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype{6} = 'T';
    
    %Now set up contrasts...
    SPMest = load('SPM.mat');
    SPMest = SPMest.SPM;
    
    SPMest.xCon = [];
    for i = 1:size(cname,2)
        if length(SPMest.xCon)==0
            SPMest.xCon = spm_FcUtil('Set',cname{i},ctype{i},'c',cons{i}',SPMest.xX.xKXs);
        else
            SPMest.xCon (end+1) = spm_FcUtil('Set',cname{i}, ctype{i},'c',cons{i}',SPMest.xX.xKXs);
        end
    end
    
    spm_contrasts(SPMest);
    
    cd(cwd);
    
end
