% Clear the command window, workspace, and close all open figures
clc
clear
close all

% Load data from a file named 'OASBUD_DATA.mat'
load('OASBUD_DATA.mat')

labels = OASBUD.class;

for Patient_Number = 1:100

    label = labels(Patient_Number);

    if label == 0
        label = 'benign';
    else
        label = 'malignant';
    end

    for Plane = 1:2

        % Create a figure to display the raw data
        figure()
        % Display the image using the y and z axis data from the loaded file
        img = imagesc(OASBUD.yaxis{Patient_Number,Plane}, OASBUD.zaxis{Patient_Number,Plane}, OASBUD.data{Patient_Number,Plane});
        
        axis image % Maintain the aspect ratio for the image
        axis off;
        
        colormap(gray); % Set the colormap to grayscale
        
        filename = "./OASBUD/" + label + "/"+label+" (patient " + string(Patient_Number)+" plane " + string(Plane) + ").png";
        exportgraphics(gcf,filename)
        
        % Create a figure to display the region of interest (ROI) data
        figure()
        % Display the ROI using the y and z axis data from the loaded file
        imagesc(OASBUD.yaxis{Patient_Number,Plane}, OASBUD.zaxis{Patient_Number,Plane}, OASBUD.roi{Patient_Number,Plane});
        axis image % Maintain the aspect ratio for the image
        axis off;
        colormap(gray); % Set the colormap to grayscale
        
        filename = "./OASBUD/" + label + "/"+label+" (patient " + string(Patient_Number)+" plane " + string(Plane) + ")_mask.png";
        exportgraphics(gcf,filename)
        close all
    end
end



