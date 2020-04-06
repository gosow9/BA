clear all; close all, clc;

% define vectors
m_error_20 = [];
mtx_20 = [];
r_dist_20 = [];

m_error_22 = [];
mtx_22 = [];
r_dist_22 = [];
t_dist_22 = [];

m_error_30 = [];
mtx_30 = [];
r_dist_30 = [];

m_error_32 = [];
mtx_32 = [];
r_dist_32 = [];
t_dist_32 = [];

% run code 10 times
for i = 1:10

    % get a random set of 40 images
    imageFileNames = [];

    for j = 1:40
        f = dir("../im/*png");
        n = numel(f);
        idx = randi(n);
        imageFileNames = [imageFileNames, strcat("../im/",f(idx).name)];       
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

    % Calibrate the camera using 2 radial distortion coefficients
    [cameraParams_20, imagesUsed_20, estimationErrors_20] = estimateCameraParameters(imagePoints, worldPoints, ...
        'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
        'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
        'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
        'ImageSize', [mrows, ncols]);
    
    % Calibrate the camera using 2 radial distortion coefficients and
    % tangential distortion
    [cameraParams_22, imagesUsed_22, estimationErrors_22] = estimateCameraParameters(imagePoints, worldPoints, ...
        'EstimateSkew', false, 'EstimateTangentialDistortion', true, ...
        'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
        'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
        'ImageSize', [mrows, ncols]);
    
    % Calibrate the camera using 3 radial distortion coefficients
    [cameraParams_30, imagesUsed_30, estimationErrors_30] = estimateCameraParameters(imagePoints, worldPoints, ...
        'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
        'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'millimeters', ...
        'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
        'ImageSize', [mrows, ncols]);
       
    % Calibrate the camera using 3 radial distortion coefficients and
    % tangential distortion
    [cameraParams_32, imagesUsed_32, estimationErrors_32] = estimateCameraParameters(imagePoints, worldPoints, ...
        'EstimateSkew', false, 'EstimateTangentialDistortion', true, ...
        'NumRadialDistortionCoefficients', 3, 'WorldUnits', 'millimeters', ...
        'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
        'ImageSize', [mrows, ncols]);
    
    % update vectors
    m_error_20 = [m_error_20; cameraParams_20.MeanReprojectionError];
    mtx_20 = [mtx_20; cameraParams_20.IntrinsicMatrix'];
    r_dist_20 = [r_dist_20; cameraParams_20.RadialDistortion];
    
    m_error_22 = [m_error_22; cameraParams_22.MeanReprojectionError];
    mtx_22 = [mtx_22; cameraParams_22.IntrinsicMatrix'];
    r_dist_22 = [r_dist_22; cameraParams_22.RadialDistortion];
    t_dist_22 = [t_dist_22; cameraParams_22.TangentialDistortion]; 
    
    m_error_30 = [m_error_30; cameraParams_30.MeanReprojectionError];
    mtx_30 = [mtx_30; cameraParams_30.IntrinsicMatrix'];
    r_dist_30 = [r_dist_30; cameraParams_30.RadialDistortion];
    
    m_error_32 = [m_error_32; cameraParams_32.MeanReprojectionError];
    mtx_32 = [mtx_32; cameraParams_32.IntrinsicMatrix'];
    r_dist_32 = [r_dist_32; cameraParams_32.RadialDistortion];
    t_dist_32 = [t_dist_32; cameraParams_32.TangentialDistortion];
end

% save vectors
save('random_result/m_error_20.txt','m_error_20','-ascii');
save('random_result/mtx_20.txt','mtx_20','-ascii');
save('random_result/r_dist_20.txt','r_dist_20','-ascii');

save('random_result/m_error_22.txt','m_error_22','-ascii');
save('random_result/mtx_22.txt','mtx_22','-ascii');
save('random_result/r_dist_22.txt','r_dist_22','-ascii');
save('random_result/t_dist_22.txt','t_dist_22','-ascii');

save('random_result/m_error_30.txt','m_error_30','-ascii');
save('random_result/mtx_30.txt','mtx_30','-ascii');
save('random_result/r_dist_30.txt','r_dist_30','-ascii');

save('random_result/m_error_32.txt','m_error_32','-ascii');
save('random_result/mtx_32.txt','mtx_32','-ascii');
save('random_result/r_dist_32.txt','r_dist_32','-ascii');
save('random_result/t_dist_32.txt','t_dist_32','-ascii');
