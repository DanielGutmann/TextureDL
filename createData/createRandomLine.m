function [ line ] = createRandomLine( linetexel,spacingVariation, imageWidth,texelHeight )
%create a line with random spacing  by repeating the texel

texel = randomizeHeight( linetexel,spacingVariation,texelHeight);
r = randi(spacingVariation);
blank = ones(size(texel,1),r );
line1 = [blank texel];
while size(line1,2) < imageWidth -  (spacingVariation + size(texel,2))
    r = randi(spacingVariation);
    blank = ones(size(texel,1),r );
    line1 = [line1 blank];
    texel = randomizeHeight( linetexel,spacingVariation,texelHeight );
    line1 =[line1  texel];
      
end
blank = ones(size(texel,1),  imageWidth - size(line1,2));
line1 = [line1 blank];
line = line1;

end

