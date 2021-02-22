#ifndef POPPY_H
#define POPPY_H

//Set this to 0 to disable the whole Poppy, both for stack tracing and performance measurement
#define STACK_TRACING_ENABLED 1

//Set this to 0 to disable performance measurement, without affecting the stack tracing
#define PERFORMANCE_COUNTING_ENABLED 0

//the interval step used for measuring performance
#define PERFORMANCE_COUNTING_INTERVAL_MS 5000 /*in milliseconds*/


#include <string>
#include <vector>
#include <list>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <map>

#ifdef __APPLE__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

using namespace std;

class CallTree;


typedef std::map<string, CallTree*> CallTreeChildrenContainer;

//returns the time in milliseconds passed since the start of the epoch
//TODO: this only works on Linux-like operating systems, port it to others. This may help: https://people.gnome.org/~fejj/code/zentimer.h
inline double CurrentTime(){
	struct timespec ts;
	return 1.0;
}

class Scope;
class CallTree{
	friend class Scope;
	private:
	double totalMsSpentHereInCurrentInterval;
	double totalMsSpentHereInPreviousInterval;
	double totalMsSpentHereSinceLaunch;
	inline void MoveToNextInterval(){
		totalMsSpentHereInPreviousInterval = totalMsSpentHereInCurrentInterval;
		totalMsSpentHereInCurrentInterval = 0;
		for(CallTreeChildrenContainer::iterator it = children.begin(); 
			it != children.end(); 
			++it){
			it->second->MoveToNextInterval();
		}
	}
	
	string name;
	CallTreeChildrenContainer children;
	CallTree* parent;
	mutable vector<CallTree*> sortedChildrenCache;
	
	static inline CallTreeChildrenContainer& GetRoots(){
		static CallTreeChildrenContainer roots;
		return roots;
	}
	
	static inline vector<CallTree*>& GetSortedRootsCache(){
		static vector<CallTree*> cache;
		return cache;
	}

	inline CallTree* GetChild(const string& name){
		//if such a child doesn't exist, insert it
		if(children.find(name) == children.end()){
			CallTree* tree = new CallTree();
			tree->parent = this;
			tree->name = name;
			children[name] = tree;
			return tree;
		}
		return children[name];
	}
	
	inline CallTree* GetParent()const{
		return parent;
	}
	
	static inline CallTree* GetRoot(const string& name){
		//if such a root doesn't exist, insert it
		if(GetRoots().find(name) == GetRoots().end()){
			CallTree* tree = new CallTree();
			tree->name = name;
			GetRoots()[name] = tree;
			return tree;
		}
		return GetRoots()[name];
	}
	
	inline void IncrementTime(double dtMs){
		totalMsSpentHereInCurrentInterval += dtMs;
		totalMsSpentHereSinceLaunch += dtMs;
	}
	
	inline static bool& IsIntervalReportReady(){
		static bool isIntervalReportReady = false;
		return isIntervalReportReady;
	}
	
	inline static void Update(double currentTime){
		static double currentIntervalStartTime = CurrentTime();
		
		//if the interval has passed, traverse the call tree and set 
		//the current interval's data as previous (ready for displaying) 
		//and reset the counters for the current interval
		if(currentIntervalStartTime + PERFORMANCE_COUNTING_INTERVAL_MS <= currentTime){
			for(CallTreeChildrenContainer::iterator it = GetRoots().begin(); 
				it != GetRoots().end(); 
				++it){
				it->second->MoveToNextInterval();
			}
			
			currentIntervalStartTime = currentTime;
			IsIntervalReportReady() = true;
		}
	}
	
	
	public:
	
	static inline string GetPerformanceReportForLastInterval(){
		#if PERFORMANCE_COUNTING_ENABLED
		if(IsIntervalReportReady()){
			ostringstream prefix;
			prefix << "Times spent in the last " << std::dec << std::setprecision(3) << PERFORMANCE_COUNTING_INTERVAL_MS/1000.0 << " seconds, as percent of parent and in milliseconds:";
			return GetPerformanceReport<double, &CallTree::totalMsSpentHereInPreviousInterval>(prefix.str());
		}
		else{
			ostringstream result;
			result << "Still gathering readings for the first interval, the interval length is " << std::dec << std::setprecision(3) << PERFORMANCE_COUNTING_INTERVAL_MS/1000.0 << " seconds...";
			return result.str();
		}
		#else
		return "Performance counting is disabled. Please set the PERFORMANCE_COUNTING_ENABLED macro to 1 to enable it.";
		#endif
	}
	
	static inline string GetPerformanceReportSinceLaunch(){
		#if PERFORMANCE_COUNTING_ENABLED
		return GetPerformanceReport<double, &CallTree::totalMsSpentHereSinceLaunch>("Times spent since launch, as percent of parent and in milliseconds:");
		#else
		return "Performance counting is disabled. Please set the PERFORMANCE_COUNTING_ENABLED macro to 1 to enable it.";
		#endif
	}
	
	private:
	template<class FieldType, FieldType CallTree::*FieldPtr>
	static string GetPerformanceReport(const string& prefix){
		
		//progress the time once more just before the report
		CallTree::Update(CurrentTime());
		
		ostringstream result;
		
		result << prefix;
			
		//sort the roots by time
		GetSortedRootsCache().clear();
		GetSortedRootsCache().reserve(GetRoots().size());
		for(CallTreeChildrenContainer::const_iterator it = GetRoots().begin(); 
			it != GetRoots().end(); 
			++it){
			GetSortedRootsCache().push_back(it->second);
		}
		sort(GetSortedRootsCache().begin(), 
			GetSortedRootsCache().end(), 
			FieldComparator<FieldType, FieldPtr>());
			
		//print each of the roots
		for(size_t i = 0; i < GetSortedRootsCache().size(); ++i){
			CallTree* tree = GetSortedRootsCache()[i];
			result << "\n------------ Call Tree #" << (i+1) << ": -----------------";
			tree->GetCallTreePerfReport<FieldType, FieldPtr>(result, 0);
		}
		return result.str();
	}
	
	template<class FieldType, FieldType CallTree::*FieldPtr>
	void GetCallTreePerfReport(ostringstream& result, int indentation)const{
		result << "\n";
		for(int i = 0; i < indentation; i++){
			result << "  ";
		}
		result << name << ": ";
		
		//get the time in milliseconds and as a percentage of the parent time
		FieldType metric = this->*FieldPtr;
		float percentOfParent = 100.0f;
		if(parent){
			if(parent->*FieldPtr != 0.0){
				percentOfParent = 100.0f * this->*FieldPtr / parent->*FieldPtr;
			}
			else{//prevent NaNs during deletion if the parent is at zero too
				percentOfParent = 0.0f;
			}
		}
		
		//output the readings
		result << std::dec << std::setprecision(3) << percentOfParent;
		result << "%, ";
		result << std::dec << std::setprecision(5) << metric;
		result << "ms";
		
		//sort the children by time
		sortedChildrenCache.clear();
		sortedChildrenCache.reserve(children.size());
		for(CallTreeChildrenContainer::const_iterator it = children.begin(); 
			it != children.end(); 
			++it){
			sortedChildrenCache.push_back(it->second);
		}
		sort(sortedChildrenCache.begin(), 
			sortedChildrenCache.end(), 
			FieldComparator<FieldType, FieldPtr>());
		
		//print each of the sorted children
		for(vector<CallTree*>::const_iterator it = sortedChildrenCache.begin(); 
			it != sortedChildrenCache.end(); 
			++it){
			(*it)->GetCallTreePerfReport<FieldType, FieldPtr>(result, indentation+1);
		}
	}
	
	//a functor for comparing two CallTrees by a given templated field of theirs
	template<class FieldType, FieldType CallTree::*FieldPtr>
	struct FieldComparator{
		bool operator()(CallTree* first, CallTree* second) const{
			return first->*FieldPtr > second->*FieldPtr;
		}
	};
};

class Scope;

enum FrameType{
	Function,
	Block,
	Section,
	Value
};
//Represents a frame in the gathered stack. 
class Frame{
	public:
	string name;
	FrameType type;
	
	//the owner of this frame
	Scope* scope;
	
	#if PERFORMANCE_COUNTING_ENABLED
	//The time (in milliseconds since app launch) when this stack frame started.
	double startTime;
	
	//The CallTree used for performance measurement associated with this frame.
	CallTree* callTree;
	#endif
	
};

//This singleton keeps the gathered trace and can print it.
//It is also used to keep the static stack data,
//so that it can be declared in a header file, without needing 
//a .cpp file definition. 
class Stack{
	friend class Scope;
	//TODO: the current implementation is not thread safe. You can make it thread-safe
	//by making the static variables thread-local
	private: list<Frame*> trace;

	//optimize the stack keeping by using a pool of reusable 
	//objects for the stack frames. This minimizes memory 
	//allocations and allows stack keeping in production builds
	private: list<Frame*> frameCache;
	
	//Let's say if we have a routine with several subfunctions and we don't place a STACK* macro 
	//in some subfunction that throws an exception. The trace that we would get is that of the 
	//last STACK-marked subfunction in the same routine, which is misleading. So after we successfully
	//exit a subfunction we leave its frame in place and raise the isLastFrameExitMarker flag, which tells us that we exited
	//this subfunction already and are beyond it. It informs us to look for the exception location further ahead.
	private: bool isLastFrameExitMarker;
	
	#if PERFORMANCE_COUNTING_ENABLED
	//the current call tree we are in. Used for measuring performance
	private: CallTree* currentCall;
	#endif
	
	private: Stack(){}
	
	public: static inline Stack* Get(){
		static Stack singleton;
		return &singleton;
	}
	
	public: static string GetTraceString(){
		//reserve the total stack length first, to reduce memory allocations
		string result = "";
		int totalLength = 0;
		int stackSize = 0;
		for(list<Frame*>::iterator it = Get()->trace.begin(); it != Get()->trace.end(); ++it){
			totalLength += (*it)->name.size() + 1 + 4;//one more for the new line char, 4 more for indentation
			stackSize++;
		}
		totalLength += 50;//some more space for the exit marker prefix
		result.reserve(totalLength);
		
		//traverse all frames and print them. Place a prefix before the last one,
		//depending on whether it is an exit marker
		int currentFrame = 0;
		for(list<Frame*>::iterator it = Get()->trace.begin(); it != Get()->trace.end(); ++it){
			if(currentFrame == stackSize - 1){
				if(Get()->isLastFrameExitMarker){
					result.append("after EXITING the scope of: ");
					result.append((*it)->name);
					result.append("\n");
				}
				else{
					result.append("while INSIDE the scope of: ");
					result.append((*it)->name);
					result.append("\n");
				}
			}
			else{
				result.append("    ");
				result.append((*it)->name);
				result.append("\n");
			}
			currentFrame++;
		}
		return result;
	}
};

//Represents a scope in the code - everything between a { and } bracket pair in a function or control block.
class Scope{
	
	//the Frame associated with this Scope object. The Frame may be returned to the reuse cache (i.e. pop)
	//before this Stack object is destroyed if it is a Section-type frame
	private: Frame* frame;
	
	private: inline Frame* GetReusableFrame(Scope* scope, const string& text, FrameType type
		#if PERFORMANCE_COUNTING_ENABLED
			, double startTime, CallTree* aCallTree
		#endif
		){
		Frame* reusableFrame;
		if(Stack::Get()->frameCache.empty()){
			reusableFrame = new Frame();
		}
		else{
			reusableFrame = Stack::Get()->frameCache.back();
			Stack::Get()->frameCache.pop_back();
		}
		reusableFrame->name = text;
		reusableFrame->type = type;
		reusableFrame->scope = scope;
		#if PERFORMANCE_COUNTING_ENABLED
			reusableFrame->startTime = startTime;
			reusableFrame->callTree = aCallTree;
		#endif
		return reusableFrame;
	}
	
	private: inline void ReturnTopFrameToCache(){
		//a Section frame might already be popped and cached if another Section has been pushed to the stack
		Frame* frameToReturn = Stack::Get()->trace.back();
		Stack::Get()->trace.pop_back();
		frameToReturn->scope = NULL; //don't risk a dangling pointer
		Stack::Get()->frameCache.push_back(frameToReturn);
	}
	private: inline void ReturnFrameToCache(list<Frame*>::reverse_iterator& frameIter){
		//only normal forward iterators work with list.erase. Reverse iterators 
		//have to be converted via base(), after being incremented (the base points to
		//the element after the reverse iterator's)
		Frame* frameToReturn = *frameIter;
		++frameIter;
		Stack::Get()->trace.erase(frameIter.base());
		frameToReturn->scope = NULL; //don't risk a dangling pointer
		Stack::Get()->frameCache.push_back(frameToReturn);
	}
	
	
	public: Scope(const string& text, FrameType type){

		//get the exit marker, if any, out of the way
		if(Stack::Get()->isLastFrameExitMarker){
			ReturnTopFrameToCache();
			Stack::Get()->isLastFrameExitMarker = false;
		}
		
		if(type == Section){
			//search for the last Section in the stack, if any, in the current
			//Function or Block and delete it. Preserve any Values you meet along the way
			list<Frame*>::reverse_iterator frameIter = Stack::Get()->trace.rbegin();
			while(frameIter != Stack::Get()->trace.rend() &&
					(*frameIter)->type != Function &&
					(*frameIter)->type != Block){
				
				if((*frameIter)->type == Value){
					++frameIter;
				}
				else{ //if(previousFrame->type == Section)
					#if PERFORMANCE_COUNTING_ENABLED
						double elapsedTime = CurrentTime() - (*frameIter)->startTime;
						(*frameIter)->callTree->IncrementTime(elapsedTime);
						//sever the link from the scope to frame, as the frame will already be recycled when the scope is destroyed

						Stack::Get()->currentCall = Stack::Get()->currentCall->GetParent();
					#endif
					
					(*frameIter)->scope->frame = NULL;
					ReturnFrameToCache(frameIter);
					//there can only be one older section, nothing more to do
					break;
				}
			}
		}
		
		#if PERFORMANCE_COUNTING_ENABLED
			double startTime;
			//Value frames don't need to measure performance
			if(type == Value){
				startTime = 0.0;
				//keep the currentCall as it is
			}
			else{
				startTime = CurrentTime();
				if(Stack::Get()->currentCall){
					Stack::Get()->currentCall = Stack::Get()->currentCall->GetChild(text);
				}
				else{
					Stack::Get()->currentCall = CallTree::GetRoot(text);
				}
				//progress the time intervals on each new scope
				CallTree::Update(startTime);
			}
		#endif
		
		frame = GetReusableFrame(this, text, type
		#if PERFORMANCE_COUNTING_ENABLED
			, startTime, Stack::Get()->currentCall
		#endif
		);
		
		//place the new frame onto the stack
		Stack::Get()->trace.push_back(frame);
	}
	
	public: virtual ~Scope(){
		//std::uncaught_exception returns true if there is an ongoing propagating exception that is unwinding the call stack.
		//Do not destroy the gathered stack in that case as we will need to print it
		if(!std::uncaught_exception()){
			
			//Section-type frames may get popped and returned to the cache before the Scope object is destroyed
			//if another Section is created in the same block
			if(frame){
				#if PERFORMANCE_COUNTING_ENABLED
					//Value frames don't need to measure performance
					if(frame->type != Value){
						double elapsedTime = CurrentTime() - frame->startTime;
						frame->callTree->IncrementTime(elapsedTime);
						CallTree* parent = Stack::Get()->currentCall->GetParent();
						Stack::Get()->currentCall = parent;
					}
				#endif
				
				//when exiting several nested scopes in a row, clear the previous exit marker
				if(Stack::Get()->isLastFrameExitMarker){
					ReturnTopFrameToCache();
				}
		
				//mark that we have succesfully exited the current top frame in the stack
				Stack::Get()->isLastFrameExitMarker = true;
			}
		}
	}
	
	
};

#define CONCATENATE_DETAIL(x, y) x##y 
#define CONCAT(x, y) CONCATENATE_DETAIL(x, y)

//extracts the filename from a path
static string extractFileName(const string& path) {
	string str(path);
	size_t slashPos = str.find_last_of("/\\");
	if(slashPos != string::npos){
		return str.substr(slashPos + 1);
	}
	else{
		return str;
	}
}

#if STACK_TRACING_ENABLED
	#define LINE_(x) #x
	#define LINE__(x) LINE_(x)
	#define LINE___ LINE__(__LINE__)

	//Marks a function in code to be added to the stack trace.
	//__FUNCTION__ is a GCC compiler-specific macro. If it doesn't work on other compilers, try __func__ and __PRETTY_FUNCTION__
	#define STACK Scope a(string("function ") + __FUNCTION__ + " at " + extractFileName(__FILE__) + ":" + LINE___, Function); 

	//Marks nested blocks, e.g. loops and "if"s , which you can also name. Several can be used inside a single block too,
	//because each will have a unique name. You can have multiple blocks in a single scope and they will stack one on top of the other, unlike sections.
	//The __COUNTER__ macro returns an auto-incrementing integer every time it is called
	#define STACK_BLOCK(x) Scope CONCAT(debugVar,__COUNTER__)(string("block \"")+#x+"\"", Block); 
	
	//Similar the STACK_BLOCK, a Section stack frame pops any previous sections in the scope of the last Block or Function. 
	//Thus sections, unlike blocks, do not nest/stack and the stack trace doesn't unnecessarily show any past sections 
	//that lead to the current position
	#define STACK_SECTION(x) Scope CONCAT(debugVar,__COUNTER__)(string("section \"")+#x+"\"", Section); 
	
	//use this to output values in the stack trace, e.g. parameter values, variables or loop counters.
	//The given value must be a string, or string expression
	#define STACK_VAL(var, value) Scope CONCAT(debugVar,__COUNTER__)(string("")+#var+" = "+(value), Value); 
#else
	#define STACK
	#define STACK_BLOCK(x)
	#define STACK_SECTION(x)
	#define STACK_VAL(var, value)
#endif

//Poppy smart. Poppy good. Poppy help master find bug.

#endif
