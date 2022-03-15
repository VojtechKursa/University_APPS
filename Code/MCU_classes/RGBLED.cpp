#include "RGBLED.h"


RGBLED::RGBLED(LED *LedR, LED *LedG, LED *LedB)
{
	this->LedR = LedR;
	this->LedG = LedG;
	this->LedB = LedB;
}


void RGBLED::SetR(int value)
{
	LedR->SetBrightness(value);
}

void RGBLED::SetG(int value)
{
	LedG->SetBrightness(value);
}

void RGBLED::SetB(int value)
{
	LedB->SetBrightness(value);
}


int RGBLED::GetR()
{
	return LedR->GetBrightness();
}

int RGBLED::GetG()
{
	return LedG->GetBrightness();
}

int RGBLED::GetB()
{
	return LedB->GetBrightness();
}


void RGBLED::SetColor(int r, int g, int b)
{
	LedR->SetBrightness(r);
	LedG->SetBrightness(g);
	LedB->SetBrightness(b);
}


void RGBLED::Tick(int step)
{
	LedR->Tick(step);
	LedG->Tick(step);
	LedB->Tick(step);
}
