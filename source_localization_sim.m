clear all
close all
clc

sound_speed = 340;

%% positions of receivers in robot, x, y, z, in meter
rec_pos1 = [0.05, 0, 1.6];
rec_pos2 = [-0.05, 0, 1.6];
rec_pos3 = [0, 0, 1.58];
rec_pos4 = [0.05, 0.05, 1.58];
rec_pos5 = [-0.1, -0.1, 1.58];

rec_pos_all = [rec_pos1; rec_pos2; rec_pos3]; %% this will affect the final prediction performance!! At least 2 receivers.
num_rec = size(rec_pos_all, 1);
%% WAV file
% Read audio signal from a WAV file
[audio_signal, fs] = audioread('0426 YJ ROOFTOP Mastered.wav');
% rec_pos1_signal = audio_signal(:, 1); % Left channel
% rec_pos2_signal = audio_signal(:, 2); % Right channel

%% signal parameters
fs = 1 * 1e6;  %% sampling rate for ADC: this will affect the final prediction performance!!
t = 0: 1/fs: 0.1;
freq_sound = 400;  %% sound frequency
noise_db = -30;  %% this will affect the final prediction performance!!
N = length(t);

%% ADC
num_bits = 16;
num_levels = 2^num_bits-1;      % number of levels: odd


%% positions of speaker
source_pos_base = [0, 2, 1.7];  % base position
source_pos_all = [];
count = 0;
for x = -1:0.5:1
    for y = -1:0.5:1
        for z = -0.5:0.5:0.5
            source_pos_all(count+1, :) = source_pos_base + [x, y, z];
            count = count + 1;
        end
    end
end


%% for each source position, generate 10 groups of features: time delay between receivers
num_trials = 10;  
Data = [];
Class = [];
num_samples = 0;
for k = 1 : size(source_pos_all, 1)
    source_pos = source_pos_all(k, :);

    for l = 1 : num_trials
        %% received signal at each receiver
        rec_signal = [];
        noise = wgn(1, N, noise_db);
        
        for i = 1 : num_rec
            rec_pos = rec_pos_all(i,:);
            dis = sqrt(sum((rec_pos - source_pos).^2));
            time_delay = dis/sound_speed;
            num_samples_delay = ceil(time_delay * fs);
            time_delay_all(i) = time_delay;
%             rec_signal(:, i) = sin(2 * pi * freq_sound * max(t - time_delay, 0)) + noise;
%             rec_signal(:, i) = audio_signal+noise;
            y = sin(2 * pi * freq_sound * max(t - i/fs, 0));  %% mimic the ADC sampling, switch into different channel after one step
            real_signal = [zeros(1, num_samples_delay) y(1:N-num_samples_delay)] + noise; 
            sampled_signal = (2/num_levels) *  round( (num_levels/2) * real_signal ) ;
%              rec_signal(:, i) = sampled_signal;
              rec_signal(:, i) = [zeros(num_samples_delay, 1); audio_signal(1:10000-num_samples_delay, 1)];
%             
        end







    
        
%          figure; plot( rec_signal(:, 1));
            % figure; plot( audio_signal(:, 1));
    
        %% compute the lagged correlation between different receivers
        features = [];
        count = 0;
        for i = 1 : num_rec
            for j = (i+1) : num_rec
                [c,lags] = xcorr(rec_signal(:,i), rec_signal(:,j));
%                 [c,lags] = xcorr(audio_signal(:, i), audio_signal(:, i));
                % stem(lags, c)
                [maxv, max_ind] = max(c);
                lag = lags(max_ind);
                features(1, count+1) = lag;
                count = count + 1;
            end
        end
        Data(num_samples+1, :) = features;
        % trainingPos(num_samples+1,:) = source_pos;
        Class(num_samples+1,:) = k;
        num_samples = num_samples + 1;
    end
end

%% split the data into two parts: training data to train the ML model, and the test data to evaluate the model performance
class_all = unique(Class);
num_class = length(class_all);
train_per = 0.8;  %% the percent of data for training, and the rest for testing
trainData = []; trainClass = [];
testData = []; testClass = [];
for i = 1 : num_class
    [c,v] = find(Class == class_all(i));
    num_data = length(c);
    random_index = randperm(num_data);
    num_training = ceil(num_data * train_per);
    idx_training = c( random_index(1: num_training) );
    idx_testing = c( random_index(num_training+1:end) );
    
    trainData = [trainData; Data(idx_training, :)];
    trainClass = [trainClass; Class(idx_training, :)];

    testData = [testData; Data(idx_testing, :)];
    testClass = [testClass; Class(idx_testing, :)];
end

%% train a classifier 
knn_classifier = fitcknn(trainData,trainClass,'NumNeighbors',5);
prediction = predict(knn_classifier, testData);
test_acc = sum(prediction == testClass) / numel(testClass)

%% get the confusion matrix: this can tell the number of data are missclassified to which class
class_all = unique(testClass);
cv_acc = zeros(length(class_all), length(class_all));
for i = 1 : length(class_all)
    [c,v] = find(testClass == class_all(i));
    num_data_class = length(c);
    pre = prediction(c,1);
    for j = 1 : length(class_all)
        [c,v] = find(pre == class_all(j));
        cv_acc(i, j) = length(c);
    end
end


