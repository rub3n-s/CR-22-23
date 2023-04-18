function output = trainFunction_d(imgPath,netPath) 

clc;

% Carregar a rede previamente treinada
load(netPath, 'net');

% Obtem o input
in = binarizedImageApp(imgPath, 0);

% Simular
out = sim(net, in);

% b - index de cada array
[a, b] = max(out);

switch(b)
    case 1
        fprintf("Resultado:  0\n")
        output = "0";
    case 2
        fprintf("Resultado:  1\n")
        output = "1";
    case 3
        fprintf("Resultado:  2\n")
        output = "2";
    case 4
        fprintf("Resultado:  3\n")
        output = "3";
    case 5
        fprintf("Resultado:  4\n")
        output = "4";
    case 6
        fprintf("Resultado: 5\n")
        output = "5";
    case 7
        fprintf("Resultado: 6\n")
        output = "6";
    case 8
        fprintf("Resultado: 7\n")
        output = "7";
    case 9
        fprintf("Resultado: 8\n")
        output = "8";
    case 10
        fprintf("Resultado: 9\n")
        output = "9";
    case 11
        fprintf("Resultado: +\n")
        output = "+";
    case 12
        fprintf("Resultado: /\n")
        output = "/";
   case 13
        fprintf("Resultado: *\n")
        output = "*";
   case 14
        fprintf("Resultado: -\n")
        output = "-";
    otherwise
        fprintf("Could not recognize\n")
        output = "Could not recognize";
end
end