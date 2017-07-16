function [ im ] = createDataset( orientation,imageHeight,imageWidth)
    orientation1rad  =   orientation * (pi/180);
    index = 1;
    numberOfTexels = 700;
    im = zeros(imageHeight,imageWidth,100);
    for texelwidth = 8:2:10
        for texelheight = 8:2:10
            for fgGrayValue =130:10:250
                for thickness = 1:3
                    fwrite(1,sprintf('%d %d \n',texelwidth,texelheight));
                    texel = createTexel(texelheight,texelwidth,orientation1rad ,fgGrayValue,0,thickness);
                    im(:,:,index) = createSyntheticImageRandomSpacing(imageWidth,imageHeight,texel,numberOfTexels);
                    index = index + 1;
                end
            end
        end
    end
end

