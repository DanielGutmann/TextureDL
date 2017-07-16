function [ im,label ] = chooseImageAndLabel( p,index,imv,imh,im45,im135 )
    if p <  0.25
        im = imv(:,:,index);
        label = zeros(size(im));
    elseif p >= 0.25 && p < 0.5
        im = imh(:,:,index);
        label = 85 * ones(size(im));
    elseif p >= 0.5 && p <= 0.75
        im = im45(:,:,index);
        label = 170 * ones(size(im));
    else
        im = im135(:,:,index);
        label = 255 * ones(size(im));
    end

end

