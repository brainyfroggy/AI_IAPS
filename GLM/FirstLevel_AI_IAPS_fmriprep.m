clear

datapath = 'N:\Experimental_Data\yujunchen\projects\AI_IAPS\GLM\betas_fmriprep';
subjects = {'Sub1','Sub2','Sub4','Sub5','Sub6','Sub7', 'Sub8', 'Sub9', 'Sub11', 'Sub12'};
% subjects = {'Sub4'};
nsub = length(subjects);
for s=1:nsub
    
    output_dir = fullfile(datapath,subjects{s});
    cd(output_dir);
    cwd=pwd;
    mov=[0 0 0 0 0 0];
    
    nsess =10;
    mns=[0 0 0 0 0 0 0 0 0 0];% 10 runs
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
    
    cname{5} = 'Up-Pl';
    simp = [ -1 0 1 0 0 0];
    cons{5} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype{5} = 'T';
  
    cname{6} = 'Up-Nt';
    simp = [ 0 -1 1 0 0 0];
    cons{6} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype{6} = 'T';
    
    cname{7} = 'AI_Pl-AI_Nt';
    simp = [ 0 0 0 1 -1 0];
    cons{7} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype{7} = 'T';
    
    cname{8} = 'AI_Up-AI_Pl';
    simp = [0 0 0 -1 0 1];
    cons{8} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype{8} = 'T';
  
    cname{9} = 'AI_Up-AI_Nt';
    simp = [ 0 0 0 0 -1 1];
    cons{9} = [simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov simp mov mns];
    ctype {9} = 'T';


%     nsess = 4;
%     mns=[0 0 0 0 ];% 2 runs
%         %main effect
%     cname{1} = 'Pl-AI_Pl';
%     simp = [ 1 0 0 -1 0 0];
%     cons{1} = [simp mov simp mov simp mov simp mov mns];
%     ctype{1} = 'T';
%     
%     cname{2} = 'Nt-AI_Nt';
%     simp = [ 0 1 0 0 -1 0];
%     cons{2} = [simp mov simp mov simp mov simp mov mns];
%     ctype{2} = 'T';
%   
%     cname{3} = 'Up-AI_Up';
%     simp = [ 0 0 1 0 0 -1];
%     cons{3} = [simp mov simp mov simp mov simp mov mns];
%     ctype{3} = 'T';
%     
%     cname{4} = 'Pl-Nt';
%     simp = [ 1 -1 0 0 0 0];
%     cons{4} = [simp mov simp mov simp mov simp mov mns];
%     ctype{4} = 'T';
%     
%     cname{5} = 'Up-Pl';
%     simp = [ -1 0 1 0 0 0];
%     cons{5} = [simp mov simp mov simp mov simp mov mns];
%     ctype{5} = 'T';
%   
%     cname{6} = 'Up-Nt';
%     simp = [ 0 -1 1 0 0 0];
%     cons{6} = [simp mov simp mov simp mov simp mov mns];
%     ctype{6} = 'T';
%     
%     cname{7} = 'AI_Pl-AI_Nt';
%     simp = [ 0 0 0 1 -1 0];
%     cons{7} = [simp mov simp mov simp mov simp mov mns];
%     ctype{7} = 'T';
%     
%     cname{8} = 'AI_Up-AI_Pl';
%     simp = [0 0 0 -1 0 1];
%     cons{8} = [simp mov simp mov simp mov simp mov mns];
%     ctype{8} = 'T';
%   
%     cname{9} = 'AI_Up-AI_Nt';
%     simp = [ 0 0 0 0 -1 1];
%     cons{9} = [simp mov simp mov simp mov simp mov mns];
%     ctype {9} = 'T';
    
    %Now set up contrasts...
    SPMest = load('SPM_estimated.mat');
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
    
end
