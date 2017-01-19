function [ texel ] = randomizeHeight( inputTexel,spacingVariation,height )
%Add blank for random height above and below the texel
r = randi(spacingVariation); 
blank = ones( r,size(inputTexel,2));
texel = [blank ; inputTexel];


texel = [texel; ones(height - size(texel,1),size(inputTexel,2)) ];


end

