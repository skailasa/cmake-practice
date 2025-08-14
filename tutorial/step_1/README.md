# 1. Basic Usage

- The major thing to remember in this tutorial is that we can use CMAKE to populate variables that are available in the code, through the `.in` header file

- project bin dir refers to the location we're building from, same kind of definition for src dir

- PRIVATE/INTERFACE/PUBLIC keyword in target include directories? What are they for? They control who gets to see this include path.
    - PRIVATE - means this is only for *this* target
    - PUBLIC - means for me and my users
    - INTERFACE - Not used for compiling this target, it's for external targets - i.e. only propageted to targets that link against it explicitly.

