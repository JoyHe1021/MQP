clear all
close all
clc

sound_speed = 340;

%% positions of receivers in robot, x, y, z, in meter
rec_pos1 = [0.05, 0, 1.6];
rec_pos2 = [-0.05, 0, 1.6];
rec_pos3 = [0, 0, 1.58];
rec_pos_all = [rec_pos1; rec_pos2; rec_pos3]; %% this will affect the final prediction performance!! At least 2 receivers.
num_rec = size(rec_pos_all, 1);

% Read audio signal from a WAV file (Assuming a stereo audio with 2 channels)
[audio_signal, fs] = audioread('0426 YJ ROOFTOP Mastered.wav');
% audio_signal = audio_signal(:, 1); % Use only the left channel

%% Signal parameters
fs = 1 * 1e6;  %% Sampling rate for ADC: this will affect the final prediction performance!!
t = 0: 1/fs: 0.1;
freq_sound = 400;  %% Sound frequency
noise_db = -30;  %% Noise level (this will affect the final prediction performance!!)
N = length(t);

%% ADC parameters
num_bits = 16;
num_levels = 2^num_bits - 1; % Number of ADC levels

%% Positions of speakers
source_pos_base = [0, 2, 1.7]; % Base position
source_pos_all = [];
count = 0;
for x = -1:0.5:1
    for y = -1:0.5:1
        for z = -0.5:0.5:0.5
            source_pos_all(count + 1, :) = source_pos_base + [x, y, z];
            count = count + 1;
        end
    end
end

%% Generate features: time delay between receivers
num_trials = 10;  
Data = [];
Class = [];
num_samples = 0;

for k = 1 : size(source_pos_all, 1)
    source_pos = source_pos_all(k, :);
    
    for l = 1 : num_trials
        %% Received signal at each receiver
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
%             1
        end


    
        %% Compute the lagged correlation between different receivers
        features = [];
        count = 0;
        for i = 1 : num_rec
            for j = (i + 1) : num_rec
                [c, lags] = xcorr(rec_signal(:, i), rec_signal(:, j));
                [maxv, max_ind] = max(c);
                lag = lags(max_ind);
                features(1, count + 1) = lag;
                count = count + 1;
            end
        end
        Data(num_samples + 1, :) = features;
        Class(num_samples + 1, :) = k;
        num_samples = num_samples + 1;
    end
end

%% Split the data into training and testing sets
class_all = unique(Class);
num_class = length(class_all);
train_per = 0.8;  %% this is the percent of data for training, and the rest for testing
trainData = []; trainClass = [];
testData = []; testClass = [];

for i = 1 : num_class
    [c, v] = find(Class == class_all(i));
    num_data = length(c);
    random_index = randperm(num_data);
    num_training = ceil(num_data * train_per);
    idx_training = c(random_index(1: num_training));
    idx_testing = c(random_index(num_training + 1: end));
    
    trainData = [trainData; Data(idx_training, :)];
    trainClass = [trainClass; Class(idx_training, :)];

    testData = [testData; Data(idx_testing, :)];
    testClass = [testClass; Class(idx_testing, :)];
end

%% Train a decision tree classifier
decision_tree_classifier = fitctree(trainData, trainClass);
% predict 
prediction_dt = predict(decision_tree_classifier, testData);
% then evaluate accuracy
test_acc_dt = sum(prediction_dt == testClass) / numel(testClass);
disp(['Decision Tree Test Accuracy: ', num2str(test_acc_dt)]);

%  confusion matrix
confusion_matrix = confusionmat(testClass, prediction_dt);
%disp('Confusion Matrix:');
%disp(confusion_matrix);





