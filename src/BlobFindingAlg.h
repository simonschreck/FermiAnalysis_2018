#ifndef u_short
#define u_short unsigned short
#endif

//just another template for creating linked lists

//Node object contains the stored object, it does inserting (always at the front, so you get a backwards arranged list)
//empty lists contains NULL pointers, end is marked with NULL pointer

template <class T>
class Node
{
public:
    Node() {myObject = 0; myNext = 0;}
    Node(T * theObject): myObject(theObject) {myNext = 0;}
    Node(T * theObject, Node<T> * theNext): myObject(theObject), myNext(theNext) {}
    ~Node() {if (myNext != 0) delete myNext; if (myObject != 0) delete myObject; myNext = 0; myObject = 0;}
    Node<T> *  Insert (T * theObject)
    {
        if (myObject != 0) return new Node<T> (theObject, this);
        else {myObject = theObject; return this;}
    }
    Node<T> * GetNext () const {return myNext;}
    T * GetThingPointer() const {return myObject;}
	
private:
    T * myObject;
    Node<T> * myNext;
};

//this class is used in the program, it creates the nodes and passes inserting and such tasks to the node
//it provides access to the stored objects and the number of objects (number convention is same as with arrays:
//two objects in a list are two objects in a list, but indices start at zero, so that the last object index is one
//
//Attention:
//it is not checked, if you give object indices bigger than the number of elements in the list, to improve speed
//so it is your task to check this

template <class T>
class LinkedList
{
public:
    LinkedList() {myFirst = new Node<T>();}
    LinkedList(T * theObject) {myFirst = new Node<T>(theObject);} 
    ~LinkedList() {delete myFirst; myFirst = 0;}
    void Insert (T * theObject)  {myFirst = myFirst->Insert (theObject);}
    int GetNumberOfObjects ();
    T * GetThing (int objectIndex);
    
private:
    Node<T> * myFirst;
};

//#include "ListTemplate.cpp"
//class that stores just the position of something as two floats
//a list of positions is the output of the whole algorithm

class BlobConfig
{
public:
    BlobConfig(int w, int h, int s, int t, int c): width(w), height(h), size(s), threshold(t), clustersize(c) {spectrum = new u_short[size];}
    const int width;
    const int height;
    const int size;
    const int threshold;
    const int clustersize;
    u_short * spectrum;
    ~BlobConfig() {delete[] spectrum;}
};

class Position
{
public:
    Position (float xPosition = 0, float yPosition = 0, float integral = 0, int area = 0): myXPosition(xPosition), myYPosition(yPosition), myIntegral(integral), myArea(area) {}
    Position (Position * thePosition): myXPosition( thePosition->GetXPosition() ), myYPosition( thePosition->GetYPosition() ), myIntegral( thePosition->GetIntegral() ), myArea( thePosition->GetArea() ) {}
    ~Position () {}
    float GetXPosition () const {return myXPosition;}
    float GetYPosition () const {return myYPosition;}
    float GetIntegral () const {return myIntegral;}
    int GetArea () const {return myArea;}
    void SetXPosition (float xPosition) {myXPosition = xPosition;}
    void SetYPosition (float yPosition) {myYPosition = yPosition;}
    void SetIntegral (float integral) {myIntegral = integral;}
    void SetArea (int area) {myArea = area;}
    
private:
    float myXPosition, myYPosition, myIntegral;
    int myArea;
};

//a pixel is the main element of an image
//it has an int, that marks its position (just an index in an array),
//it stores its z-value since it is set to zero while searching blobs, so that it can be recreated
//it can check its four neighbours, if they are pixels of the same blob and
//it remebers having checked the neighbours
class Pixel
{
public:
    Pixel (int position, const BlobConfig * bc): myPosition(position), myNeighboursChecked(false), mybc(bc)
    {myZValue = mybc->spectrum[position];}
    ~Pixel () { }
    float GetZValue () const {return myZValue;}
    int GetPosition () const {return myPosition;}
    void ResetZValue () { mybc->spectrum[myPosition] =  myZValue; }
    bool CheckAllNeighbours (LinkedList<Pixel> & pixelList);
    bool GetNeighboursChecked () {return myNeighboursChecked;}
    
private:
    int myPosition;
    bool myNeighboursChecked;
    u_short myZValue;
    const BlobConfig * mybc;
    
    bool CheckNeighbour (LinkedList<Pixel> & pixelList, int anotherPosition);
};

//a blob is what the algorithm looks for:
//some pixels with z-values above the threshold, that are four-connected
//that means that they have shared sides, not only corners
//constructor declares an empty list for the pixels in a blob,
//sets first pixel to zero, so that it will not be identified as new blob
class Blob
{
public:
    Blob (int firstPixel, const BlobConfig * bc): myArea(1), myAreaFound(false), mybc(bc), myPixelList(new Pixel(firstPixel, mybc)), myPosition(new Position()) {mybc->spectrum[firstPixel] =  0;}
    ~Blob () {delete myPosition;}
    Position * GetPosition ();
    int FindArea();
    int GetArea();
    void RecreateArea();
    
private:
    int myArea;
    bool myAreaFound;
    const BlobConfig * mybc;
    LinkedList<Pixel> myPixelList;
    Position * myPosition;
};

//this is the main class, that has full ability in BlobFinding
//its main function takes an empty list of positions (of blobs) and returns it filled
class BlobFindingMain
{
public:
    BlobFindingMain (BlobConfig * bc): mybc(bc) {}
    ~BlobFindingMain () {}
    int GetBlobPositions (LinkedList<Position> *, bool recreateArea = false);
    
private:
    int * GetBlobPositionsPart (LinkedList<Position> *, int startPixel, int endPixel, bool recreateArea);
    const BlobConfig * mybc;
};
