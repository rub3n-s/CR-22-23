function [in] =  binarizedImageInput(imgPath, inMatrix)
%% Conceito da Funcao
% Criar uma matriz binaria a partir de uma imagem recebida por parametro
% e devolver essa matriz binaria como input

%% Definir Constantes e Variaveis
% Resolucao das imagens
% Tamanho padrao das imagens 150x150
% Minimo 25x25  
IMG_RES = [25 25]; 

%% Ler, redimensionar as imagens e preparar os targets
binaryMatrix = zeros(IMG_RES(1) * IMG_RES(2), 1);
targetMatrix = [];

if(imgPath ~= "null")
    img = imread(sprintf(imgPath));
    img=im2gray(img);
    img = imresize(img, IMG_RES);
else
    img = imresize(inMatrix, IMG_RES);
end


binarizedImg = imbinarize(img);
binaryMatrix(:, 1) = reshape(binarizedImg, 1, []);

in = binaryMatrix;
end