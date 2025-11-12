"""
@file   : has_cycle.py
@time   : 2025-11-12
"""
class Solution:
    def hasCycle(self, head) -> bool:
        if head is None or head.next is None:
            return False
        slow = head
        fast = head.next  #
        while slow != fast:
            slow = slow.next

            if fast is not None and fast.next is not None:
                fast = fast.next.next
            else:
                break

        if fast is None or fast.next is None:
            return False
        else:
            return True

