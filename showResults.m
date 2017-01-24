
numImages = 5;
for i = 1:numImages
    filename = strcat('C:\Users\iiitb 2012\Documents\GitHub\TextureDL\data\im\im',num2str(i),'.jpg');
    im = imread(filename);
    resultFileName = strcat('C:\Users\iiitb 2012\Documents\GitHub\TextureDL\data\output\im',num2str(i),'.jpg');
    result = imread(resultFileName);

    subplot(numImages,2,2 * i -1);imshow(im);
    subplot(numImages,2,2 * i);imshow(result);
end
