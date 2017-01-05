%create a synthetic image with single textured region
%Put texels at random points

imageWidth = 200;
imageHeight = 400;
texelwidth = 8;
texelheight = 8;

im = ones(imageWidth,imageHeight);
numberOfTexels = 1150;
numberOfpointsAdded = 0;
while(numberOfpointsAdded < numberOfTexels)
    x = randi(imageWidth);
    y = randi(imageHeight);
    
    %check if (x,y) is near boundary
    if(x - texelwidth/2  + 1 > 1 && x + texelwidth/2 < imageWidth && y - texelheight/2 + 1 > 1 && y + texelheight/2 <imageHeight)
        %check for overlapping texels
        window = im(x - texelwidth/2 + 1 :x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 );
        if(min(min(window)) ==1)
            texel = createTexel1(texelheight,texelwidth,3*(pi/4),1,0);
            im(x - texelwidth/2 + 1:x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 ) = texel;
            numberOfpointsAdded = numberOfpointsAdded + 1;
        end
        
    end
    
    
end


im1 = ones(imageWidth,imageHeight);
numberOfTexels = 1100;
numberOfpointsAdded = 0;
while(numberOfpointsAdded < numberOfTexels)
    x = randi(imageWidth);
    y = randi(imageHeight);
    
    %check if (x,y) is near boundary
    if(x - texelwidth/2  + 1 > 1 && x + texelwidth/2 < imageWidth && y - texelheight/2 + 1 > 1 && y + texelheight/2 <imageHeight)
        %check for overlapping texels
        window = im1(x - texelwidth/2 + 1 :x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 );
        if(min(min(window)) ==1)
            texel = createTexel1(texelheight,texelwidth,0,1);
            im1(x - texelwidth/2 + 1:x + texelwidth/2,y - texelheight/2 + 1: y + texelheight/2 ) = texel;
            numberOfpointsAdded = numberOfpointsAdded + 1;
        end
        
    end
    
    
end

im = [im;im1]
fileName ='C:\TextureDL\Data\RandomSpacingTwoTexton1.jpg';
imwrite(im,fileName,'jpg');

imshow(im);


