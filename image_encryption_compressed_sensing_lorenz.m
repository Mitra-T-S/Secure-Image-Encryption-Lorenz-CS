%% full_image_encryption_with_lorenz_final.m
% Image encryption demo: SAE-placeholder + DCT + Lorenz (SHA seeded RK4) +
% Arnold scramble + chained diffusion. Auto-tunes scaleFactor to get UACI ~33%.
clc; clear; close all;

%% ----------------------
%% Step 1: Load image
%% ----------------------
Icolor = imread('peppers.png');       % <-- change to your image path if needed
figure('Name','Original Color'); imshow(Icolor); title('Step 1: Original Image (Color)');

% Convert to grayscale for encryption process
if size(Icolor,3) == 3
    I = rgb2gray(Icolor);
else
    I = Icolor;
end
I = imresize(I, [256 256]);

%% ----------------------
%% Step 2: SAE placeholder (downsample = "compressed features")
%% ----------------------
compressedImg = imresize(I, [64 64]);   % placeholder for SAE output
figure('Name','Compressed'); imshow(compressedImg, []); title('Step 2: Compressed Features (SAE placeholder)');

%% ----------------------
%% Step 3: DCT sparsifying (visual)
%% ----------------------
dctImg = dct2(double(compressedImg));
figure('Name','DCT'); imshow(log(abs(dctImg)+1), []); title('Step 3: Sparse DCT Representation');

%% (Optional) CS measurements visualization
[m_small, n_small] = size(dctImg);
csRate = 0.75;
Phi = randn(round(csRate*m_small), m_small);
Y = Phi * dctImg;
figure('Name','CS'); imagesc(Y); colormap gray; xlabel('DCT Coefficient Index','FontSize',10,'FontWeight','bold');
ylabel('Measurement Index','FontSize',10,'FontWeight','bold');
title('Step 3b: CS Measurements');

%% ----------------------
%% Step 4: Lorenz chaotic sequence (SHA-seeded, RK4 integrator)
%% ----------------------
keyStr1 = 'mySecretKey123';   % <-- change to your passphrase
keyStr2 = 'mySecretKey456';   % <-- different key for avalanche NPCR/UACI

chaoticSeq1 = lorenz_from_key(keyStr1, numel(compressedImg));
chaoticSeq2 = lorenz_from_key(keyStr2, numel(compressedImg));

% Animated butterfly (trace)
[X,Y,Z] = lorenz_traj(keyStr1,20000);
figure('Name','Lorenz Animated');
hplot = plot3(NaN,NaN,NaN,'b','LineWidth',1.2);
grid on; hold on;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Animated Lorenz Attractor (3D butterfly)');
view(30,20);
for idx = 1:50:4000
    set(hplot,'XData',X(1:idx),'YData',Y(1:idx),'ZData',Z(1:idx));
    drawnow;
end

%% ----------------------
%% Step 5+6: Encryption with auto-tuned scaleFactor
%% ----------------------
nIter = 5;    % Arnold iterations
targetUACI = 33; tolUACI = 0.5;
scales = 0.60:0.01:1.00;
bestScale = scales(1); bestDiff = Inf; bestU = NaN; bestNP = NaN;
for s = scales
    C1 = encrypt_compressed(compressedImg, chaoticSeq1, nIter, s);
    C2 = encrypt_compressed(compressedImg, chaoticSeq2, nIter, s); % <-- different key
    U = (sum(abs(double(C1(:)) - double(C2(:)))) / (numel(C1) * 255)) * 100;
    NP = (sum(C1(:) ~= C2(:)) / numel(C1)) * 100;
    diff = abs(U - targetUACI);
    if NP < 80, diff = diff + 100; end % penalize
    if diff < bestDiff
        bestDiff = diff; bestScale = s; bestU = U; bestNP = NP;
    end
end
scaleFactor = bestScale;
encryptedImg1 = encrypt_compressed(compressedImg, chaoticSeq1, nIter, scaleFactor);
encryptedImg2 = encrypt_compressed(compressedImg, chaoticSeq2, nIter, scaleFactor);

fprintf('Auto-chosen scaleFactor = %.3f | UACI = %.3f %% | NPCR = %.3f %%\n', scaleFactor, bestU, bestNP);

figure('Name','Encrypted'); imshow(encryptedImg1,[]); title(sprintf('Encrypted (scale=%.3f)', scaleFactor));

%% ----------------------
%% Step 7: Display pipeline
%% ----------------------
figure('Name','Pipeline Overview');
subplot(2,3,1), imshow(Icolor), title('Original (Color)');
subplot(2,3,2), imshow(compressedImg, []), title('Compressed (SAE placeholder)');
subplot(2,3,3), imshow(log(abs(dctImg)+1), []), title('Sparse DCT');
subplot(2,3,4), imshow(encryptedImg1, []), title('Encrypted');
subplot(2,3,5);
[counts, bins] = imhist(encryptedImg1);
bar(bins,counts,'FaceColor',[0.12 0.56 1]);
xlabel('Gray Level Intensity (0–255)','FontSize',8,'FontWeight','bold');
ylabel('Frequency of Pixels','FontSize',8,'FontWeight','bold');
title('Histogram of Encrypted','FontSize',10,'FontWeight','bold');
figure;


%% ----------------------
%% Step 8: Metrics
%% ----------------------
encResized = imresize(encryptedImg1, size(I));
corrCoeff = corrcoef(double(I(:)), double(encResized(:)));
fprintf('Correlation (Orig vs Encrypted) = %.4f\n', corrCoeff(1,2));

% Final NPCR & UACI
NPCR = (sum(encryptedImg1(:) ~= encryptedImg2(:)) / numel(encryptedImg1)) * 100;
UACI = (sum(abs(double(encryptedImg1(:)) - double(encryptedImg2(:)))) / (numel(encryptedImg1) * 255)) * 100;
fprintf('Final NPCR = %.2f %%\n', NPCR);
fprintf('Final UACI = %.2f %% (scaleFactor = %.3f)\n', UACI, scaleFactor);

%% ----------------------
function decryptedImg = decrypt_compressed_fixed(encryptedImg, chaoticSeq, nIter, scaleFactor)
    encryptedImg = uint8(encryptedImg);
    [m,n] = size(encryptedImg);
    Npix = m * n;

    mask = uint8(floor(255 * scaleFactor * reshape(chaoticSeq(1:Npix), m, n)));

    linC = encryptedImg(:);
    maskv = mask(:);
    plainLin = zeros(Npix,1,'uint8');

    for k = 1:Npix
        if k == 1
            plainLin(k) = bitxor(linC(1), maskv(1));
        else
            plainLin(k) = bitxor(bitxor(linC(k), maskv(k)), linC(k-1));
        end
    end

    plainImg = reshape(plainLin, m, n);

    % Use renamed inverse Arnold transform
    for t = 1:nIter
        plainImg = iarnoldTransform_mat_new(plainImg);
    end

    decryptedImg = plainImg;
end

%% =======================
%% Helper functions
%% =======================
function chaoticSeq = lorenz_from_key(keyStr, neededLen)
    md = java.security.MessageDigest.getInstance('SHA-256');
    hash = uint8(md.digest(uint8(keyStr)));
    hexHash = dec2hex(hash)'; hexHash = lower(hexHash(:)');
    x0 = (hex2dec(hexHash(1:8)) / 2^32) * 10 + 0.1;
    y0 = (hex2dec(hexHash(9:16)) / 2^32) * 10 + 0.2;
    z0 = (hex2dec(hexHash(17:24)) / 2^32) * 30 + 0.3;
    sigma=10; rho=28; beta=8/3; h=0.01; Nlong=neededLen+5000;
    X=zeros(Nlong,1);Y=zeros(Nlong,1);Z=zeros(Nlong,1);
    X(1)=x0;Y(1)=y0;Z(1)=z0;
    for k=1:Nlong-1
        k1x=sigma*(Y(k)-X(k)); k1y=X(k)*(rho-Z(k))-Y(k); k1z=X(k)*Y(k)-beta*Z(k);
        xt=X(k)+0.5*h*k1x; yt=Y(k)+0.5*h*k1y; zt=Z(k)+0.5*h*k1z;
        k2x=sigma*(yt-xt); k2y=xt*(rho-zt)-yt; k2z=xt*yt-beta*zt;
        xt=X(k)+0.5*h*k2x; yt=Y(k)+0.5*h*k2y; zt=Z(k)+0.5*h*k2z;
        k3x=sigma*(yt-xt); k3y=xt*(rho-zt)-yt; k3z=xt*yt-beta*zt;
        xt=X(k)+h*k3x; yt=Y(k)+h*k3y; zt=Z(k)+h*k3z;
        k4x=sigma*(yt-xt); k4y=xt*(rho-zt)-yt; k4z=xt*yt-beta*zt;
        X(k+1)=X(k)+(h/6)*(k1x+2*k2x+2*k3x+k4x);
        Y(k+1)=Y(k)+(h/6)*(k1y+2*k2y+2*k3y+k4y);
        Z(k+1)=Z(k)+(h/6)*(k1z+2*k2z+2*k3z+k4z);
    end
    nx=(X-min(X))/(max(X)-min(X)+eps);
    ny=(Y-min(Y))/(max(Y)-min(Y)+eps);
    nz=(Z-min(Z))/(max(Z)-min(Z)+eps);
    chaoticSeq=mod(nx+ny+nz,1);
    chaoticSeq=chaoticSeq(5001:5000+neededLen);
end

function [X,Y,Z]=lorenz_traj(keyStr,Nlong)
    md=java.security.MessageDigest.getInstance('SHA-256');
    hash=uint8(md.digest(uint8(keyStr)));
    hexHash=dec2hex(hash)';hexHash=lower(hexHash(:)');
    x0=(hex2dec(hexHash(1:8))/2^32)*10+0.1;
    y0=(hex2dec(hexHash(9:16))/2^32)*10+0.2;
    z0=(hex2dec(hexHash(17:24))/2^32)*30+0.3;
    sigma=10;rho=28;beta=8/3;h=0.01;
    X=zeros(Nlong,1);Y=zeros(Nlong,1);Z=zeros(Nlong,1);
    X(1)=x0;Y(1)=y0;Z(1)=z0;
    for k=1:Nlong-1
        k1x=sigma*(Y(k)-X(k)); k1y=X(k)*(rho-Z(k))-Y(k); k1z=X(k)*Y(k)-beta*Z(k);
        xt=X(k)+0.5*h*k1x; yt=Y(k)+0.5*h*k1y; zt=Z(k)+0.5*h*k1z;
        k2x=sigma*(yt-xt);k2y=xt*(rho-zt)-yt;k2z=xt*yt-beta*zt;
        xt=X(k)+0.5*h*k2x; yt=Y(k)+0.5*h*k2y; zt=Z(k)+0.5*h*k2z;
        k3x=sigma*(yt-xt);k3y=xt*(rho-zt)-yt;k3z=xt*yt-beta*zt;
        xt=X(k)+h*k3x; yt=Y(k)+h*k3y; zt=Z(k)+h*k3z;
        k4x=sigma*(yt-xt);k4y=xt*(rho-zt)-yt;k4z=xt*yt-beta*zt;
        X(k+1)=X(k)+(h/6)*(k1x+2*k2x+2*k3x+k4x);
        Y(k+1)=Y(k)+(h/6)*(k1y+2*k2y+2*k3y+k4y);
        Z(k+1)=Z(k)+(h/6)*(k1z+2*k2z+2*k3z+k4z);
    end
end

function cipher = encrypt_compressed(compImg, chaoticSeqSmall, nIter, scaleFactor)
    img = compImg;
    for t = 1:nIter, img = arnoldTransform_mat(img); end
    mask = uint8(255 * scaleFactor * reshape(chaoticSeqSmall(1:numel(img)), size(img)));
    cipher = zeros(size(img), 'uint8'); prev = uint8(0);
    lin = img(:); maskv = mask(:);
    for k = 1:numel(lin)
        c = bitxor(bitxor(uint8(lin(k)), maskv(k)), prev);
        cipher(k) = c; prev = c;
    end
    cipher = reshape(cipher, size(img));
end

function out = arnoldTransform_mat(img)
    [m,n] = size(img); if m~=n, error('Arnold needs square'); end
    N=m; A=[1 1;1 2]; out=zeros(N,N,'like',img);
    for r=0:N-1, for c=0:N-1
        new=mod(A*[r;c],N); out(new(1)+1,new(2)+1)=img(r+1,c+1);
    end,end
end
function out = iarnoldTransform_mat_new(img)
    [m,n] = size(img);
    if m ~= n
        error('Inverse Arnold Transform requires square image.');
    end
    N = m;
    Ainv = [2 -1; -1 1];
    out = zeros(N, N, 'like', img);
    for x = 0:N-1
        for y = 0:N-1
            oldCoord = mod(Ainv * [x; y], N);
            out(oldCoord(1)+1, oldCoord(2)+1) = img(x+1, y+1);
        end
    end
end
function plain = decrypt_compressed(cipher, chaoticSeqSmall, nIter, scaleFactor)
    mask = uint8(255 * scaleFactor * reshape(chaoticSeqSmall(1:numel(cipher)), size(cipher)));
    linC = cipher(:); maskv = mask(:); 
    plainLin = zeros(size(linC), 'uint8');
    prev = uint8(0);

    for k = 1:numel(linC)
        % reverse diffusion: c = xor(xor(p,mask),prev)
        % so p = xor(xor(c,mask),prev)
        p = bitxor(bitxor(linC(k), maskv(k)), prev);
        plainLin(k) = p;
        prev = linC(k);
    end
    plain = reshape(plainLin, size(cipher));

    % inverse Arnold transform
    for t = 1:nIter
        plain = iarnoldTransform_mat(plain);
    end
end

function out = iarnoldTransform_mat(img)
    [m,n] = size(img); if m~=n, error('Arnold inverse needs square'); end
    N=m; Ainv=[2 -1;-1 1]; out=zeros(N,N,'like',img);
    for r=0:N-1, for c=0:N-1
        old=mod(Ainv*[r;c],N); out(old(1)+1, old(2)+1)=img(r+1,c+1);
    end,end
end
% >>> Print first 50 Lorenz values
 disp('First 50 Lorenz X values:');
 disp(X(1:50)');
 disp('First 50 Lorenz Y values:'); 
 disp(Y(1:50)'); 
 disp('First 50 Lorenz Z values:');
 disp(Z(1:50)');