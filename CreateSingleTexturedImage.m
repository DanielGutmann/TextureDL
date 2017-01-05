clc;
close all;
clear all;

folderName = 'sig';
orientation1degrees = 45;
orientation1rad  =   orientation1degrees * (pi/180);


index = 1;

%TODO grayscale

numberOfTexels = 900;
for texelwidth = 10:2:16
    for texelheight = 10:2:16
        for fgGrayValue =128:255
            if index == 405
                disp index;
            end
            
           
            texel = createTexel2(texelheight,texelwidth,orientation1rad ,fgGrayValue,0);
            imageWidth = 400;
            imageHeight = 400;

            im = createSyntheticImageRandomSpacing(imageWidth,imageHeight,texel,numberOfTexels);
            %imshow(im);

            fileName =strcat('C:\TextureDL\data\',folderName,'\im', int2str(index),'.jpg');
            imwrite(im,fileName,'jpg');
            index = index + 1;
        end
    end
end
