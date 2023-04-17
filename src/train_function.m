function train_function(op)

fprintf('\n***************************************\n')
fprintf(' A iniciar o treino para o digito [%s]\n',op)
fprintf('***************************************\n\n')

%% Definir Constantes
% Resolucao das imagens
% Tamanho padrao das imagens 150x150
% Minimo 25x25
IMG_RES = [25 25];

% alinea a)
% Caminho para a pasta
FOLDER_PATH = sprintf('NN_datasets/start/%s/',op);

% Numero de ficheiros de imagem por pasta
NUM_FILES = 5;

% alinea b)
% Caminho para a pasta
%FOLDER_PATH = sprintf('NN_datasets/train/%s/',op);

% Numero de ficheiros de imagem por pasta
%NUM_FILES = 50;

%% Ler, redimensionar e preparar os targets
numBW = zeros(IMG_RES(1) * IMG_RES(2), NUM_FILES);

% Ler os 5 elementos dentro da pasta
for i=1:NUM_FILES
    img = imread(strcat(FOLDER_PATH,sprintf('%d.png',i)));
    img = imresize(img,IMG_RES);
    binarizedImg = imbinarize(img);
    numBW(:, i) = reshape(binarizedImg, 1, []);
end

%% Target
numTarget = [eye(NUM_FILES)];

%% Treinar rede
% alinea a) Testar apenas com 10 camadas escondidas
%net = feedforwardnet(10);

% alinea b) Testar com varios neuronios e camadas escondidas
net = feedforwardnet([10 10 10]);

%% Configurar a Rede
% alinea b) Testar varios parametros

% Função de Ativação
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
net.layers{3}.transferFcn = 'purelin';

% Numero de Epocas
net.trainParam.epochs = 100;

% Funcao de Treino
net.trainFcn = 'trainlm';

% Divisao de Treino
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

%% Realizar 10 iteracoes de treino e calcular media
somaTreinos = 0;

for k=1:10
    %% Treinar, Simular e Apresentar Resultados
    % Treinar 
    [net,tr] = train(net, numBW, numTarget);    

    % Simular
    out = sim(net, numBW);
    
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
    fprintf('\nPrecisao do Treino [%d] = %.2f\n', k, accuracy)

    plotconfusion(numTarget,out) % Matriz de confusao

    %% Desempenho
    %disp(tr.performFcn);
    disp(['Train Performance: ' num2str(tr.best_perf)])
    disp(['Validation Performance: ' num2str(tr.best_vperf)]);
    disp(['Test Performance: ' num2str(tr.best_tperf)]);
end

%% Apresentar a Media
fprintf('\n------ Apos 10 iteracoes ------\n')
fprintf('Media de Precisao = %.2f\n',somaTreinos/10);


end