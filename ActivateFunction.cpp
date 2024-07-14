#include "ActivateFunction.h"

void ActivateFunction::set()
{
	std::cout << "Set activate function please:\n1-sigmoid; \n2-ReLU; \n3-th(x).\n";
	int tmp;
	std::cin >> tmp;
	switch (tmp)
	{
	case sigmoid:
		actFunc = sigmoid;
		break;
	case ReLU: 
		actFunc = ReLU;
		break;
	case thx: 
		actFunc = thx;
		break;
	default:
		throw std::runtime_error("Error read activate function\n");
		break;
	}
}

void ActivateFunction::use(double* value, int n)
{
	switch(actFunc)
	{
	case activateFunc::sigmoid:
		for (int i = 0; i < n; i++)
			value[i] = 1 / (1 + exp(-value[i]));
		break;
	case activateFunc::ReLU:
		for (int i = 0; i < n; i++)
		{
			if (value[i] < 0)
				value[i] *= 0.01;
			else if (value[i] > 1)
				value[i] = 1 + 0.01 * (value[i] - 1);
			//else value = value
		}
		break;
	case activateFunc::thx:
		for (int i = 0; i < n; i++)
		{
			if (value[i] < 0)
				value[i] = 0.01 * (exp(value[i]) - exp(-value[i])) / (exp(value[i]) + exp(-value[i]));
			else if (value[i] >= 0)
				value[i] = (exp(value[i]) - exp(-value[i])) / (exp(value[i]) + exp(-value[i]));
		}
		break;
	default:
		throw std::runtime_error("Error activate function\n");
		break;
	}
}

void ActivateFunction::useDer(double* value, int n)
{
	switch (actFunc)
	{
	case activateFunc::sigmoid:
		for (int i = 0; i < n; i++)
			value[i] = value[i] * (1 - value[i]);
		break;
	case activateFunc::ReLU:
		for (int i = 0; i < n; i++)
		{
			if (value[i] < 0 || value[i]>1)
				value[i] = 0.01;
			else
				value[i] = 1;
		}
		break;
	case activateFunc::thx:
		for (int i = 0; i < n; i++)
		{
			if (value[i] < 0)
				value[i] = 0.01 * (1 - value[i] * value[i]);
			else
				value[i] = 1 - value[i] * value[i];
		}
		break;
	default:
		throw std::runtime_error("Error activate derivative of function\n");
		break;
	}
}

double ActivateFunction::useDer(double value)
{
	switch (actFunc)
	{
	case activateFunc::sigmoid:
			value = value * (1 - value);
		break;
	case activateFunc::ReLU:
			if (value < 0 || value>1)
				value = 0.01;
			else
				value = 1;
		break;
	case activateFunc::thx:
		if (value < 0)
			value = 0.01 * (1 - value * value);
		else
			value = 1 - value * value;
		break;
	default:
		throw std::runtime_error("Error activate derivative of function\n");
		break;
	}
}
