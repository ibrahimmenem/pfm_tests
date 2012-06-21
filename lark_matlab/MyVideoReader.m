clear all
close all
xyloObj = VideoReader('/home/ibrahim/pfm_tests/Videos/GB_1.MP4'); % '~/test.mp4'    '/home/ibrahim/pfm_tests/Videos/rand3.MP4'
nFrames = xyloObj.NumberOfFrames;
vidHeight = xyloObj.Height;
vidWidth = xyloObj.Width;
DSF=100;
% Preallocate movie structure.
mov(1:nFrames/DSF) = ...
    struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'),...
           'colormap', []);
i=0;
% Read one frame at a time.
for k = 1 : nFrames 
    display(k)
    if mod(k,DSF)==0 % downsampling
       i=i+1 ;
       mov(i).cdata = read(xyloObj, k);
    else
       % read(xyloObj, k);
    end
end
display('fin')

% Size a figure based on the video's width and height.
hf = figure;
set(hf, 'position', [150 150 vidWidth vidHeight])

% Play back the movie once at the video's frame rate.
 movie(hf, mov, 1, xyloObj.FrameRate);