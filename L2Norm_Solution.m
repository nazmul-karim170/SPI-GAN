
%% Load the .mat file that contains the grayscale images
% path(fullfile(pwd,'function'),path);
% dataFolder = fullfile('data','stl10_matlab');
% dataset = ["unlabeled", "test", "train"];
% fprintf(dataset(1));
% X = load(fullfile(dataFolder, strcat(dataset(1),".mat")));
% size(struct2cell(X{'X'}),1)
clear all;
clc;
close all;

%% Set the path
path(fullfile(pwd,'function'),path);

% Download the stl-10
%- This can be done manually if the gunzip or untar functions are not available.
%- http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz
% url = 'http://ai.stanford.edu/~acoates/stl10/stl10_matlab.tar.gz';
% disp('-- Loading stl-10 dataset')
% gunzip(url, 'data');
% untar(fullfile('data','stl10_matlab.tar'),'data');


% User-defined
dataFolder = fullfile('data','stl10_matlab');
saveFolder = '/home/ryota/anaconda3/Project_2021/Computational_Imaging /Main Methods/SPIRIT-master/data/STL10_reconstruction/CS_Images_unlabeled_0.40/';
testset = 'test';                       %-    8'000 images. 'train' works too (5'000 images)
trainset = 'unlabeled';                 %-      100'000 images


% compression ratio
ratio = 0.4;                           %- retain 10% of the measurements
imsize = 64;
noise_level = 1e-4;

%% Preprocess training data
X = preprocess_stl10(dataFolder, 'unlabeled', saveFolder, ratio, noise_level);

%% Function for Reconstruction
function recons_image = Cs_reconstruction_test(input_image, Phi, Theta_inv, psi, index, saveFolder, m, noise_level)

% INPUT IMAGE
im_size = size(input_image, 1);
A = squeeze(input_image);
x = double(A(:));
n = length(x);

% COMPRESSION
y = Phi*x;
% y = y + 4096*noise_level*randn(m,1);

% L2 NORM SOLUTION 
s2 = Theta_inv*y;

% IMAGE RECONSTRUCTIONS
x2 = zeros(n,1);

% disp(size(psi(1,:,:)));
for ii = 1:n
    x2 = x2+psi(ii,:,:)'*s2(ii);
end

recons_image = reshape(x2,im_size,im_size);

dataFolder1 = saveFolder;
name = strcat(strcat("recons_image_", string(index)), ".png");
image_name = fullfile(dataFolder1, name); 
imwrite(uint8(recons_image), image_name);

end


%% PREPROCESS_STL10 Preprocess the STL-10 dataset 
function X = preprocess_stl10(dataFolder, dataset, saveFolder, ratio, noise_level)


% Default arguments
if nargin<4, saveFolder = dataFolder; end
saveFolder = fullfile(saveFolder); 
dataFolder = fullfile(dataFolder);

% Load raw data
disp('-- Loading raw data')
X = load(fullfile(dataFolder, [dataset,'.mat']));
X = getfield(X, 'X');

% Preprocessing
disp('-- Preprocessing')

% Reshaping
X_im = reshape(X,[size(X,1),96,96,3]);


%-- init output
X = zeros(64,64,size(X_im,1),'uint8');
X_inter = zeros(64,64,size(X_im,4),'uint8');
clear X;

% dataFolder1 = '/home/ryota/anaconda3/Trojan_AI_Project/TRojAI_UCF/Computational Imaging /SPIRIT-master/data/CS_Input_Images_64';
for jj=1: size(X_im,1)
    
    % convert to grayscale
    X_inter = rgb2gray(squeeze(X_im(jj,:,:,:)));
    X(:,:,jj) = imresize(X_inter,[64,64]);
    
%     disp('--Saving the image');
%     name = strcat(strcat("original_image_", string(jj)), ".png");
%     image_name = fullfile(dataFolder1, name); 
%     imwrite(uint8(X(:,:,jj)), image_name);
    
end

compression_ratio = ratio;
X_recons = zeros(64,64,size(X,3),'uint8');

%___INPUT IMAGE___
im_size = size(X_recons(:,:,1), 1);
A = squeeze(X_recons(:,:,1));
% size(A)
x = double(A(:));
n = length(x);


%NUMBER OF MEASUREMENTS
m = floor(compression_ratio*n);     % NOTE: small error still present after increasing m to 1500;

% MEASUREMENT MATRIX___ 
Phi = (sign(randn(m,n))+ones(m,n))/2;
Phi = orth(Phi')'; 

%___THETA___
% NOTE: Avoid calculating Psi (nxn) directly to avoid memory issues.
Theta = zeros(m,n);
for ii = 1:n
    ek = zeros(1,n);
    ek(ii) = 1;
    psi = idct(ek)';
    Theta(:,ii) = Phi*psi;
end

Shi = zeros(n,n,1);
for ii = 1:n
    ek = zeros(1,n);
    ek(ii) = 1;
    Shi(ii,:,:) = idct(ek)';
end

Theta_inv= pinv(Theta);

% Number of images to process
for kk= 1:45000
    X_recons(:,:,kk) = Cs_reconstruction_test(X(:,:,kk), Phi,Theta_inv, Shi, kk, saveFolder, m, noise_level);
end

% Save as preprocessed images (8-bit)
disp('-- Saving preprocessed data')
fullfilename = fullfile(saveFolder,[dataset,'MNIST_30_Test.mat']);
save(fullfilename, 'X_recons');

end
