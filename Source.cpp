#include "NetWork.h"
#include <chrono>
#include <string>

struct data_info
{
    double* pixels;
    int digit;
};

void displayAsciiImage(string filename)
{
    ifstream file(filename);

    if (!file.is_open())
    {
        cerr << "Failed to open the file." << endl;
        return;
    }

    string line;
    while (getline(file, line))
    {
        cout << line << endl;
    }

    file.close();
}

data_NetWork ReadDataNetWork(string path) 
{
    data_NetWork data{};
    ifstream fin;
    fin.open(path);
    if (!fin.is_open()) 
    {
        cout << "Error reading the file " << path << endl;
        displayAsciiImage("error.txt");
        cout << endl;
        system("pause");
    }
    else
    {
        cout << path << " loading...\n";
        displayAsciiImage("loading.txt");
        cout << endl;
    }
    string tmp;
    int L;
    while (!fin.eof()) 
    {
        fin >> tmp;
        if (tmp == "NetWork") 
        {
            fin >> L;
            data.L = L;
            data.size = new int[L];
            for (int i = 0; i < L; i++) 
            {
                fin >> data.size[i];
            }
        }
    }
    fin.close();
    return data;
}

data_info* ReadData(string path, const data_NetWork& data_NW, int& examples) 
{
    data_info* data;
    ifstream fin;
    fin.open(path);
    if (!fin.is_open()) 
    {
        cout << "Error reading the file " << path << endl;
        displayAsciiImage("error.txt");
        cout << endl;
        system("pause");
    }
    else
    {
        cout << path << " loading... \n";
        displayAsciiImage("loading.txt");
        cout << endl;
    }
    string tmp;
    fin >> tmp;
    if (tmp == "Examples") 
    {
        fin >> examples;
        cout << "Examples: " << examples << endl;
        data = new data_info[examples];
        for (int i = 0; i < examples; ++i)
            data[i].pixels = new double[data_NW.size[0]];

        for (int i = 0; i < examples; ++i) 
        {
            fin >> data[i].digit;
            for (int j = 0; j < data_NW.size[0]; ++j) 
            {
                fin >> data[i].pixels[j];
            }
        }
        fin.close();
        cout << "lib_MNIST loaded... \n";
        return data;
    }
    else
    {
        cout << "Error loading: " << path << endl;
        fin.close();
        displayAsciiImage("error.txt");
        cout << endl;
        return nullptr;
    }
}

int main()
{
    NetWork NW{};
    data_NetWork NW_config;
    data_info* data;
    double ra = 0, predict, maxra = 0;
    int epoch = 0, numberOfExample=0, right;
    bool study, repeat = true;
    chrono::duration<double> time;

    NW_config = ReadDataNetWork("Config.txt");
    NW.Init(NW_config);
    NW.PrintConfig();

    while (repeat) 
    {
        cout << "Start training the NetWork? (1/0)" << endl;
        cin >> study;
        if (study)
        {
            int examples;
            data = ReadData("lib_MNIST_edit.txt", NW_config, examples);
            auto begin = chrono::steady_clock::now();
            while (ra / examples * 100 < 100)
            {
                ra = 0;
                auto t1 = chrono::steady_clock::now();
                for (int i = 0; i < examples; ++i) 
                {
                    NW.SetInput(data[i].pixels);
                    right = data[i].digit;
                    predict = NW.ForwardFeed();
                    if (predict != right) {
                        NW.BackPropogation(right);
                        NW.WeightsUpdater(0.15 * exp(-epoch / 20.));
                    }
                    else
                        ra++;
                }
                auto t2 = chrono::steady_clock::now();
                time = t2 - t1;
                if (ra > maxra) maxra = ra;
                cout << "Right answers: " << ra / examples * 100 << "\tMax right answers: " << maxra / examples * 100 << "\tEpoch: " << epoch << "\tTime: " << time.count() << endl;
                epoch++;
                if (epoch == 20)
                    break;
            }
            auto end = chrono::steady_clock::now();
            time = end - begin;
            cout << "Time: " << time.count() / 60. << " minutes" << endl;
            NW.SaveWeights();
        }
        else 
        {
            NW.ReadWeights();
        }
        cout << "Test? (1/0)\n";
        bool test_flag;
        cin >> test_flag;
        if (test_flag) 
        {
            int ex_tests;
            data_info* data_test;
            data_test = ReadData("lib_10k.txt", NW_config, ex_tests);
            ra = 0;
            for (int i = 0; i < ex_tests; ++i) 
            {
                NW.SetInput(data_test[i].pixels);
                predict = NW.ForwardFeed();
                right = data_test[i].digit;
                numberOfExample++;
                if (right == predict)
                {
                    ra++;
                    switch (right) 
                    {
                    case 0:
                        cout << "#" << numberOfExample << ") Right answer: " << "\t"; displayAsciiImage("0.txt"); cout << endl;
                        break;
                    case 1:
                        cout << "#" << numberOfExample << ") Right answer: " << "\t"; displayAsciiImage("1.txt"); cout << endl;
                        break;
                    case 2:
                        cout << "#" << numberOfExample << ") Right answer: " << "\t"; displayAsciiImage("2.txt"); cout << endl;
                        break;
                    case 3:
                        cout << "#" << numberOfExample << ") Right answer: " << "\t"; displayAsciiImage("3.txt"); cout << endl;
                        break;
                    case 4:
                        cout << "#" << numberOfExample << ") Right answer: " << "\t"; displayAsciiImage("4.txt"); cout << endl;
                        break;
                    case 5:
                        cout << "#" << numberOfExample << ") Right answer: " << "\t"; displayAsciiImage("5.txt"); cout << endl;
                        break;
                    case 6:
                        cout << "#" << numberOfExample << ") Right answer: " << "\t"; displayAsciiImage("6.txt"); cout << endl;
                        break;
                    case 7:
                        cout << "#" << numberOfExample << ") Right answer: " << "\t"; displayAsciiImage("7.txt"); cout << endl;
                        break;
                    case 8:
                        cout << "#" << numberOfExample << ") Right answer: " << "\t"; displayAsciiImage("8.txt"); cout << endl;
                        break;
                    case 9:
                        cout << "#" << numberOfExample << ") Right answer: " << "\t"; displayAsciiImage("9.txt"); cout << endl;
                        break;
                    default:
                        cout << "Invalid answer." << endl;
                        break;
                    }

                }
            }
            cout << "Percentage of correct answers: " << ra / ex_tests * 100 << endl;
        }
        epoch = 0;
        cout << "Repeat? (1/0)\n";
        cin >> repeat;
    }
    system("pause");
    return 0;
}
