# Program to implement a singly linked list

class Node:
    """
    Represents a single node in a singly linked list.
    """

    # Function to initialise a new node
    def __init__(self, data, next_node= None):
        self.data = data
        self.next = next_node


class LinkedList:
    """
    Implements a singly linked list.
    """

    # Function to initialise a new linkedlist
    def __init__(self):
        self.head = None
        self.size: int = 0

    # Function to display the linked list
    def __str__(self) -> str:
        if self.head is None:
            return "[]"

        nodes = []
        current = self.head
        while current is not None:
            nodes.append(str(current.data))
            current = current.next
        return f"[{' -> '.join(nodes)}]"

    # Function to track length of linked list
    def __len__(self) -> int:
        return self.size

    # Function to add new elements to linkedlist
    def append(self, data) -> None:
        new_node = Node(data)

        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = new_node

        self.size += 1

    # Function to delete the given node of a linkedlist
    def delete_by_index(self, index: int) -> None:
        if self.head is None:
            raise ValueError("Cannot delete from an empty list.")

        if not (1 <= index <= self.size):
            raise ValueError(f"Index {index} is out of range. List size is {self.size}.")

        # Case 1: Deleting the head node
        if index == 1:
            self.head = self.head.next
        else:
            # Case 2: Deleting any other node
            previous = self.head
            for _ in range(index - 2):
                previous = previous.next

            previous.next = previous.next.next

        self.size -= 1

# Testing the code
if __name__ == "__main__":
    ll = LinkedList()

    # Adding new elements
    print("Adding elements 20, 5, 10...")
    ll.append(20)
    ll.append(5)
    ll.append(10)

    # Initial list
    print(f"Initial list: {ll}")
    print(f"Length: {len(ll)}")

    # Deleting an element
    print("\nDeleting node at index 2 (value 5)...")
    ll.delete_by_index(2)

    # Printing list
    print(f"List after deletion: {ll}")
    print(f"New length: {len(ll)}")

    # Test error handling
    print("\nTesting error cases:")
    try:
        ll.delete_by_index(99)
    except ValueError as e:
        print(f"Successfully caught expected error: {e}")