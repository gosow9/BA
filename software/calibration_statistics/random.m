clear all; close all, clc;

m_error = [];
mtx = [];
r_dist = [];
t_dist = [];

% run code 5 times
for i = 1:5

    % get a random set of 20 images
    imageFileNames = [];

    for j = 1:20
        f = dir("im/*png");
        n = numel(f);
        idx = randi(n);
        imageFileNames = [imageFileNames, strcat("im/",f(idx).name)];       
    end

    % Detect checkerboards in images
    [imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
    imageFileNames = imageFileNames(imagesUsed);

    % Read the first image to obtain image size
    originalImage = imread(imageFileNames{1});
    [mrows, ncols, ~] = size(originalImage);

    % Generate world coordinates of the corners of the squares
    squareSize = 15;  % in units of 'millimeters'
    worldPoints = generateCheckerboardPoints(boardSize, squareSize);

    % Calibrate the camera
    [cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
        'EstimateSkew', false, 'EstimateTangentialDistortion', true, ...
        'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'millimeters', ...
        'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
        'ImageSize', [mrows, ncols]);

    m_error = [m_error; cameraParams.MeanReprojectionError];
    mtx = [mtx; cameraParams.IntrinsicMatrix'];
    r_dist = [r_dist; cameraParams.RadialDistortion];
    t_dist = [t_dist; cameraParams.TangentialDistortion];
    
end

save('random_result/m_error.txt','m_error','-ascii');
save('random_result/mtx.txt','mtx','-ascii');
save('random_result/r_dist.txt','r_dist','-ascii');
save('random_result/t_dist.txt','t_dist','-ascii');
