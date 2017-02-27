function [cropped_im, ov, jrect] = jitter_imcrop(im, rect, IoU)
% im: input whole image
% rect: [horizontal pos of top-left point, vertical pos of top-left point,
%        width of box, height of box]
% IoU: 0~1 intersection-area/union-area
    w = size(im,2);
    h = size(im,1);
    assert(rect(1) > 0);
    assert(rect(2) > 0);
    assert((rect(1) + rect(3)) <= w);
    assert((rect(2) + rect(4)) <= h);
    
    bbox_w = rect(3);
    bbox_h = rect(4);

    % original imcrop
    if IoU >= 1
        cropped_im = imcrop(im, rect);
        jrect = rect;
        ov = 1;
        return;
    end
    
    % jittered crop to make IoU >= input IoU
    assert(IoU > 0 && IoU < 1);
    jrect = rect;
    % horizontal position of top-left point
    b = (1-IoU);
    a = (1-1/IoU);
    while 1
        r = a + (b-a).*rand(4,1);
        
        jrect = [rect(1) + r(1)*bbox_w, rect(2) + r(2)*bbox_h,  rect(3) + r(3)*bbox_w, rect(4) + r(4)*bbox_h];
        jextremes = [max(min(jrect(1), w), 1), max(min(jrect(2), h), 1), max(min(jrect(1) + jrect(3), w), 1), max(min(jrect(2) + jrect(4), h), 1)];
        jextremes = round(jextremes);        
        ov = box_overlap([rect(1),rect(2),rect(1)+rect(3),rect(2)+rect(4)], jextremes);
        jrect = [jextremes(1), jextremes(2), jextremes(3) - jextremes(1), jextremes(4) - jextremes(2)];
    
        if ov > IoU
            break;
        end
    end
    cropped_im = imcrop(im, jrect);
end
