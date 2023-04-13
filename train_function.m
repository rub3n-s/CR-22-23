function train_function(op)

fprintf('\n***************************************\n')
fprintf(' A iniciar o treino para o digito [%s]\n',op)
fprintf('***************************************\n\n')

%% Definir Constantes
% Resolucao das imagens
% Tamanho das imagens 150x150
% Minimo 25x25
IMG_RES = [25 25];

% Numero de ficheiros de imagem por pasta
NUM_FILES = 5;

%% Ler, redimensionar e preparar os targets
numBW = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);

% Ler os 5 elementos dentro da pasta
for i=1:NUM_FILES
    img = imread(sprintf('NN_datasets//start//%s//%d.png',op,i));
    img = imresize(img,IMG_RES);
    binarizedImg = imbinarize(img);
    numBW(:, i) = reshape(binarizedImg, 1, []);
end

%% Target
numTarget = [eye(5)];

%% Treinar rede
net = feedforwardnet(10);

%% Configurar a Rede
%net.trainFcn = 'trainlm';
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'purelin';
%net.divideFcn = 'dividerand';
%net.divideParam.trainRatio = 1;
%net.divideParam.valRatio = 0;
%net.divideParam.testRatio = 0;

%% Realizar 10 iteracoes de treino e calcular media
somaTreinos = 0;

for k=1:10
    %% Simular e analisar resultados
    [net,tr] = train(net, numBW, numTarget);    
    out = sim(net, numBW);    
    %disp(tr);
    
    r = 0;
    for i=1:size(out,2)
        [a b] = max(out(:,i));
        [c d] = max(numTarget(:,i));
        if b == d
          r = r+1;
        end
    end
    
    accuracy = r/size(out,2);
    somaTreinos = somaTreinos + accuracy;
    fprintf('Precisao do Treino [%d] = %.2f\n', k, accuracy)
end

%% Apresentar a Media
fprintf('\n------ Apos 10 iteracoes ------\n')
fprintf('Media de Precisao = %.2f\n',somaTreinos/10);

%% Apresentar o plotconflusion
plotconfusion(numTarget,out)
end