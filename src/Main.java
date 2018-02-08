import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.TreeSet;


public class Main {
    public static void main(String[] args) {
        System.out.println("hello world123");
//        moveZeros(new int[]{2, 3, 1, 0, 2, 0, 3, 4, 5, 6, 70, 22, 0, 0, 02, 0, 12});
//        int[] a = {2, 3, 1, 2, 4, 3};
//        int[] res = twoSum2(a, 30);
//        System.out.println(minSubArrayLen(a, 7) + "");
//        removeSortedArray3(new int[]{1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 7, 7});
//        mergeSortedArray(new int[]{0, 2, 4, 5, 6, 7, 9, 11, 23}, new int[]{1, 2, 3, 5, 16, 17, 19, 111, 123});
//        System.out.println(lengthOfLongestSubstring("aaacad123sb123"));
//        printArr(twoSum3(new int[]{2, 7, 11, 1}, 9));
//        System.out.println(validAnagram("ab", "a"));
//        threeSum(new int[]{-1, 0, 1, 2, -1, -4});
//        System.out.println(maxSubArray(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4}));
//        get50thNumber();
//        int[] a = new int[]{-1, -1};
//        System.out.println(containsNearbyDuplicate(a, 1));
        int l = 10000,r=20000;
        int hh = (int)((long) l+(long)r)/2;
    }

    /**
     * LeetCode 19. Remove Nth Node From End of List
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode startNode = new ListNode(0);
        ListNode slowNode = startNode, fastNode = slowNode;
        slowNode.next = head;
        for (int i = 0; i < n + 1; i++) {
            fastNode = fastNode.next;
        }
        while (fastNode != null) {
            slowNode = slowNode.next;
            startNode = startNode.next;
        }
        slowNode.next = slowNode.next.next;
        return startNode.next;
    }

    /**
     * LeetCode 237. Delete Node in a Linked List
     */
    public void deleteNode(ListNode node) {
        if (node.next == null) {
            node = null;
        } else {
            node.val = node.next.val;
            node.next = node.next.next;
        }
    }

    /**
     * LeetCode 24. Swap Nodes in Pairs
     * https://leetcode.com/problems/swap-nodes-in-pairs/description/
     * 时间复杂度: O(n)
     * 空间复杂度: O(1)
     */
    public ListNode swapPairs(ListNode head) {
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;

        ListNode p = dummyHead;
        while (p.next != null && p.next.next != null) {
            ListNode node1 = p.next;
            ListNode node2 = node1.next;
            ListNode next = node2.next;
            node2.next = node1;
            node1.next = next;
            p.next = node2;
            p = node1;
        }

        return dummyHead.next;
    }


    public ListNode deleteDuplicates(ListNode head) {
        // 创建虚拟头结点
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;

        ListNode pre = dummyHead;
        ListNode cur = head;

        while (cur != null) {
            while (cur.next != null && cur.val == cur.next.val) {
                cur = cur.next;
            }
            if (pre.next == cur) {
                pre = pre.next;
            } else {
                pre.next = cur.next;
            }
            cur = cur.next;
        }
        return dummyHead.next;
    }


    /**
     * 203. Remove Linked List Elements
     * https://leetcode.com/problems/remove-linked-list-elements/description/
     * 使用虚拟头结点
     * 时间复杂度: O(n)
     * 空间复杂度: O(1)
     *
     * @param head
     * @param val
     * @return
     */
    public ListNode removeElements(ListNode head, int val) {
        // 创建虚拟头结点
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = head;

        ListNode cur = dummyHead;
        while (cur.next != null) {
            if (cur.next.val == val) {
                ListNode delNode = cur.next;
                cur.next = delNode.next;
            } else {
                cur = cur.next;
            }
        }
        return dummyHead.next;
    }

    private class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
        }
    }

    /**
     * LeetCode 206. Reverse Linked List 链表反转
     */
    public ListNode reverseList(ListNode head) {
        /* recursive solution */
        return reverseListInt(head, null);
    }

    private ListNode reverseListInt(ListNode head, ListNode newHead) {
        if (head == null)
            return newHead;
        ListNode next = head.next;
        head.next = newHead;
        return reverseListInt(next, head);
    }

    /**
     * 给出数组整型nums，是否存在索引i和j，是的nums[i]和nums[j]之间的差别不超过给定的整数t，且i和j之间的差别不超过给定的整数k
     *
     * @param nums
     * @param k    |i-j| <= k
     * @param t    |nums[i]-nums[j]| <=t
     * @return
     */
    private static boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
        TreeSet<Long> record = new TreeSet<>();
        for (int i = 0; i < nums.length; i++) {

            if (record.ceiling((long) nums[i] - (long) t) != null &&
                    record.ceiling((long) nums[i] - (long) t) <= (long) nums[i] + (long) t)
                return true;

            record.add((long) nums[i]);

            if (record.size() == k + 1)
                record.remove((long) nums[i - k]);
        }

        return false;
    }

    private static boolean containsNearbyDuplicate(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i]) && i - map.get(nums[i]) <= k) {
                return true;
            } else {
                map.remove(nums[i]);
                map.put(nums[i], i);
            }
        }
        return false;
    }

    /**
     * LeetCode 447. Number of Boomerangs
     * <p>
     * Given n points in the plane that are all pairwise distinct, a "boomerang" is a tuple of points (i, j, k) such that the distance between i and j equals the distance between i and k (the order of the tuple matters).
     * <p>
     * Find the number of boomerangs. You may assume that n will be at most 500 and coordinates of points are all in the range [-10000, 10000] (inclusive).
     * <p>
     * 时间复杂度: O(n^2)
     * 空间复杂度: O(n)
     *
     * @param points
     * @return
     */
    private static int numberOfBoomerangs(int[][] points) {
        int result = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < points.length; i++) {
            for (int j = 0; j < points.length; j++) {
                if (i == j)
                    continue;
                int distance = getDistance(points[i], points[j]);
                map.put(distance, map.getOrDefault(distance, 0) + 1);
            }
        }
        for (int val : map.values()) {
            result += val * (val - 1);
        }
        map.clear();

        return result;
    }


    private static int getDistance(int[] a, int[] b) {
        int dx = a[0] - b[0];
        int dy = a[1] - b[1];

        return dx * dx + dy * dy;
    }


    /**
     * 49. Group Anagrams
     * Given an array of strings, group anagrams together.
     * <p>
     * For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
     * Return:
     * [
     * ["ate", "eat","tea"],
     * ["nat","tan"],
     * ["bat"]
     * ]
     *
     * @param strs
     */
    private static List<List<String>> groupAnagrams(String[] strs) {
        if (strs == null || strs.length == 0) {
            return new ArrayList<>();
        }
        HashMap<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = String.valueOf(chars);
            if (!map.containsKey(key)) {//这个字符串不存在
                map.put(key, new ArrayList<String>());
            }
            map.get(key).add(str);
        }

        return (List<List<String>>) map.values();
    }

    /**
     * LeetCode 454. 4Sum
     *
     * @param A
     * @param B
     * @param C
     * @param D
     * @return
     */
    private static int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
        HashMap<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < C.length; i++) {
            for (int j = 0; j < D.length; j++) {
                int sum = C[i] + D[j];
                map.put(sum, map.getOrDefault(sum, 0) + 1);
            }
        }

        int res = 0;
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < B.length; j++) {
                res += map.getOrDefault(-1 * (A[i] + B[j]), 0);
            }
        }

        return res;
    }

    private static void get50thNumber() {
        Random random = new Random();
        int[] nums = new int[200000000];
        for (int i = 0; i < nums.length; i++) {
            nums[i] = random.nextInt();
        }
        Date date1 = new Date();
//        Arrays.sort(nums);
//        System.out.println(nums[nums.length - 50]);

        for (int i = 0; i < 50; i++) {
            for (int j = nums.length - 1; j > i; j--) {
                if (nums[j] > nums[j - 1]) {//交换位置
                    int temp = nums[j];
                    nums[j] = nums[j - 1];
                    nums[j - 1] = temp;
                }
            }
        }
        System.out.println(nums[49]);

        Date date2 = new Date();
        System.out.println(date2.getTime() - date1.getTime());
    }
    /*
    *
        System.out.println("begin");
    * */

    /**
     * LeetCode 53. Maximum Subarray
     *
     * @param nums
     * @return
     */
    private static int maxSubArray(int[] nums) {
        int curSum = 0;
        int maxSum = nums[0];
        for (int i = 0; i < nums.length; i++) {
            if (curSum > 0) {
                curSum += nums[i];
            } else {
                curSum = nums[i];
            }
            if (curSum > maxSum) {
                maxSum = curSum;
            }
        }
        return maxSum;
    }


    /**
     * LeetCode 15. Three Sum
     * Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.
     * <p>
     * Note: The solution set must not contain duplicate triplets.
     *
     * @param nums
     * @return
     */
    private static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> list = new LinkedList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (i == 0 || (i > 0 && nums[i] != nums[i - 1])) {
                int lo = i + 1, hi = nums.length - 1;
                while (lo < hi) {
                    if (0 == nums[lo] + nums[hi] + nums[i]) {//找到了一组数据
                        list.add(Arrays.asList(nums[i], nums[lo], nums[hi]));
                        while (lo < hi && nums[lo] == nums[lo + 1]) {
                            lo++;
                        }
                        while (lo < hi && nums[hi] == nums[hi - 1]) {
                            hi--;
                        }
                        lo++;
                        hi--;
                    } else if (nums[lo] + nums[hi] + nums[i] < 0) {//三者相加为负数
                        lo++;
                    } else {
                        hi--;
                    }
                }
            }
        }

        return list;
    }

    /**
     * LeetCode 1. Two Sum
     *
     * @param nums
     * @param target
     * @return
     */
    private static int[] twoSum3(int[] nums, int target) {
        int[] result = new int[2];
        HashMap<Integer, Integer> map = new HashMap<>();//<数字，游标>
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                result[1] = map.get(target - nums[i]);
                result[0] = i;
                return result;
            }
            map.put(nums[i], i);
        }
        return result;
    }

//作业  202, 290, 205, 451

    /**
     * LeetCode 242. Valid Anagram
     * s = "anagram", t = "nagaram", return true.
     * s = "rat", t = "car", return false.
     *
     * @return
     */
    public static boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        char[] str1 = s.toCharArray();
        char[] str2 = t.toCharArray();
        Arrays.sort(str1);
        Arrays.sort(str2);
        return Arrays.equals(str1, str2);
    }

    /**
     * LeetCode 242. Valid Anagram
     * s = "anagram", t = "nagaram", return true.
     * s = "rat", t = "car", return false.
     *
     * @return
     */
    private static boolean validAnagram(String s, String t) {
        char[] sChars = s.toCharArray();
        char[] tChars = t.toCharArray();
        HashMap<Character, Integer> record = new HashMap<>();
        for (Character c : sChars) {
            if (!record.containsKey(c)) {
                record.put(c, 1);
            } else {
                record.put(c, record.get(c) + 1);
            }
        }

        for (Character c : tChars) {
            if (!record.containsKey(c) || record.get(c) == 0) {
                return false;
            } else {
                record.put(c, record.get(c) - 1);
            }
        }

        return true;
    }


    private static int[] intersectionOfTwoArrays2(int[] nums1, int[] nums2) {
        HashMap<Integer, Integer> record = new HashMap<Integer, Integer>();
        for (int num : nums1)
            if (!record.containsKey(num))
                record.put(num, 1);
            else
                record.put(num, record.get(num) + 1);

        ArrayList<Integer> result = new ArrayList<Integer>();
        for (int num : nums2)
            if (record.containsKey(num) && record.get(num) > 0) {
                result.add(num);
                record.put(num, record.get(num) - 1);
            }

        int[] ret = new int[result.size()];
        int index = 0;
        for (Integer num : result)
            ret[index++] = num;

        return ret;
    }

    /**
     * LeetCode 349. Intersection of Two Arrays
     *
     * @param nums1
     * @param nums2
     * @return
     */
    private static int[] intersectionOfTwoArrays(int[] nums1, int[] nums2) {
        //O(n)
        HashSet<Integer> record = new HashSet<>();
        for (int num : nums1) {
            record.add(num);
        }
        //O(n)
        HashSet<Integer> resultSet = new HashSet<>();
        for (int num : nums2) {
            if (record.contains(num)) {
                resultSet.add(num);
            }
        }
        int[] res = new int[resultSet.size()];
        int index = 0;
        for (Integer num : resultSet) {
            res[index++] = num;
        }
        return res;
    }

    /**
     * LeetCode 3. Longest Substring Without Repeating Characters
     *
     * @param s
     * @return
     */
    private static int lengthOfLongestSubstring(String s) {
        char[] array = s.toCharArray();
        int[] req = new int[256];

        int maxCount = 0;
        int left = 0, right = -1;
        while (left < array.length) {
            if (right + 1 < array.length && req[(int) array[right + 1]] == 0) {
                req[(int) array[++right]]++;
            } else {
                req[(int) array[left++]]--;
            }
            if (maxCount < right - left + 1) {
                maxCount = right - left + 1;
            }
        }
        return maxCount;
    }

    /**
     * LeetCode 209. Minimum Size Subarray Sum
     *
     * @param nums
     * @param s
     * @return
     */
    private static int minSubArrayLen(int[] nums, int s) {
        int leftIndex = 0, rightIndex = -1;//nums[leftIndex, rightIndex]是滑动窗口
        int sum = 0;
        int res = nums.length + 1;//保存滑动窗口的长度
        while (leftIndex < nums.length) {
            if (rightIndex + 1 < nums.length && sum < s) {
                rightIndex++;
                sum += nums[rightIndex];
            } else {
                sum -= nums[leftIndex];
                leftIndex++;
            }
            if (sum >= s && rightIndex - leftIndex + 1 < res) {
                res = rightIndex - leftIndex + 1;
            }
        }
        if (res == nums.length + 1) {
            res = 0;
        }
        return res;
    }

    /**
     * LeetCode 11
     *
     * @param height
     * @return
     */
    public static int maxArea(int[] height) {
        int area = 0;
        int leftIndex = 0, rightIndex = height.length - 1;
        while (leftIndex < rightIndex) {
            int tempArea = height[leftIndex] < height[rightIndex] ?
                    height[leftIndex] * (rightIndex - leftIndex) : height[rightIndex] * (rightIndex - leftIndex);
            if (area < tempArea) {
                area = tempArea;
            }
            if (height[leftIndex] < height[rightIndex])
                ++leftIndex;
            else --rightIndex;
        }
        return area;
    }


    /**
     * 在numbers中找到两个数a,b使target=a+b;
     * 更高效的
     * 对撞指针
     *
     * @param numbers
     * @param target
     * @return
     */
    private static int[] twoSum2(int[] numbers, int target) {
        int i = 0, j = numbers.length - 1;
        while (i < j) {
            if (numbers[i] + numbers[j] < target) {
                i++;
            } else if (numbers[i] + numbers[j] > target) {
                j--;
            } else {
                break;
            }
        }
        return new int[]{i, j};
    }

    /**
     * 在numbers中找到两个数a,b使target=a+b;
     *
     * @param numbers
     * @param target
     * @return
     */
    private static int[] twoSum(int[] numbers, int target) {
        int[] result = new int[]{-1, -1};
        for (int leftIndex = 0; leftIndex < numbers.length; leftIndex++) {
            int rightIndex = binarySearch(numbers, leftIndex, target - numbers[leftIndex]);
            if (rightIndex != -1) {
                result[0] = leftIndex + 1;
                result[1] = rightIndex + 1;
                return result;
            }
        }
        return result;
    }

    /**
     * 二分查找法
     *
     * @param arr
     * @param target
     * @return
     */
    private static int binarySearch(int[] arr, int start, int target) {
        int numCount = arr.length;
        int leftIndex = start;
        int rightIndex = numCount - 1;//定义[leftIndex, rightIndex]中寻找target
        int i = 0;
        while (leftIndex <= rightIndex) {//当l=r时，[leftIndex, rightIndex]依然有效
            int mid = (leftIndex + rightIndex) / 2;

            if (target > arr[mid]) {
                leftIndex = mid + 1;
            } else if (target < arr[mid]) {
                rightIndex = mid - 1;
            } else if (arr[mid] == target) {
                return mid;
            }
        }
        return -1;
    }


    /**
     * 快速排序
     *
     * @param a
     * @param low
     * @param high
     */
    public static void sort(int[] a, int low, int high) {
        int start = low;
        int end = high;
        int key = a[low];


        while (end > start) {
            //从后往前比较
            while (end > start && a[end] >= key)  //如果没有比关键值小的，比较下一个，直到有比关键值小的交换位置，然后又从前往后比较
                end--;
            if (a[end] <= key) {
                int temp = a[end];
                a[end] = a[start];
                a[start] = temp;
            }
            //从前往后比较
            while (end > start && a[start] <= key)//如果没有比关键值大的，比较下一个，直到有比关键值大的交换位置
                start++;
            if (a[start] >= key) {
                int temp = a[start];
                a[start] = a[end];
                a[end] = temp;
            }
            //此时第一次循环比较结束，关键值的位置已经确定了。左边的值都比关键值小，右边的值都比关键值大，但是两边的顺序还有可能是不一样的，进行下面的递归调用
        }
        //递归
        if (start > low) sort(a, low, start - 1);//左边序列。第一个索引位置到关键值索引-1
        if (end < high) sort(a, end + 1, high);//右边序列。从关键值索引+1到最后一个
    }


    /**
     * LeetCode 215
     * 在一个整数序列中寻找第k大的元素
     * ·给定数组[3,2,1,5,6,4], k=2, 结果为5
     * ·利用快排partition中，将pivot放置在了其正确的位置上的性质
     */
    private static void LargestElementInAnArray(int[] nums) {

    }

    /**
     * LeetCode 88
     * 给定有序数组nums1,nums1，将nums2中的数字归并到nums1中，并保证nums1中数字有序
     *
     * @param nums1
     * @param nums2
     */
    private static void mergeSortedArray(int[] nums1, int[] nums2) {
        int[] nums = new int[nums1.length + nums2.length];
        int nums1Index = 0;
        int nums2Index = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums1Index < nums1.length && nums1[nums1Index] < nums2[nums2Index]) {
                nums[i] = nums1[nums1Index];
                nums1Index++;
            } else {
                nums[i] = nums2[nums2Index];
                nums2Index++;
            }
        }

        for (int i = 0; i < nums.length; i++) {
            System.out.print(nums[i] + ",");

        }
    }


    /**
     * 二分查找法
     *
     * @param arr
     * @param target
     * @return
     */
    private static int binarySearch(int[] arr, int target) {
        int numCount = arr.length;
        int leftIndex = 0;
        int rightIndex = numCount - 1;//定义[leftIndex, rightIndex]中寻找target
        int i = 0;
        while (leftIndex <= rightIndex) {//当l=r时，[leftIndex, rightIndex]依然有效
            int mid = (leftIndex + rightIndex) / 2;

            if (target > arr[mid]) {
                leftIndex = mid + 1;
            } else if (target < arr[mid]) {
                rightIndex = mid - 1;
            } else if (arr[mid] == target) {
                return mid;
            }

            System.out.println("循环第" + i++ + "次");
        }
        return -1;
    }

    /**
     * 将0移动到一边
     * O(n)
     *
     * @param nums
     */
    private static void moveZeros(int[] nums) {
        if (nums == null) {
            return;
        }
        int[] nonZerosNums = new int[nums.length];
        int j = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nonZerosNums[j] = nums[i];
                j++;
            }
        }
        for (; j < nonZerosNums.length; j++) {
            nonZerosNums[j] = 0;
        }
        for (int i = 0; i < nums.length; i++) {

            nums[i] = nonZerosNums[i];
        }
        /*for (int i = 0; i < nonZerosNums.length; i++) {
            if (i >= nonZerosNums.length - 1) {
                System.out.print(nonZerosNums[i] + "]");
            } else if (i == 0) {
                System.out.print("[" + nonZerosNums[i] + ",");
            } else {
                System.out.print(nonZerosNums[i] + ",");
            }
        }*/
    }


    /**
     * 将0移动到一边
     * O(n)
     *
     * @param nums
     */
    private static void moveZeros2(int[] nums) {
        int k = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                if (i > k) {
                    nums[k] = nums[i];
                    nums[i] = 0;
                }
                k++;//记录已经找到的非0数的个数
            }
        }

        for (int i = 0; i < nums.length; i++) {
            if (i >= nums.length - 1) {
                System.out.print(nums[i] + "]");
            } else if (i == 0) {
                System.out.print("[" + nums[i] + ",");
            } else {
                System.out.print(nums[i] + ",");
            }
        }
    }

    private static void removeElement(int[] nums, int value) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == value) {//找到了目标数字
                nums[count] = nums[i];
                count++;
            }
        }
        int[] result = new int[count];
        for (int i = 0; i < count; i++) {
            result[i] = nums[i];
        }
    }

    /**
     * 有序数组去重
     *
     * @param nums
     */
    private static void removeSortedArray(int[] nums) {
        if (nums.length == 0) {
            return;
        }
        int value = nums[0];
        int realIndex = 1;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != value) {//说明是重复
                nums[realIndex] = nums[i];
                value = nums[i];
                realIndex++;
            }
        }

        System.out.println("");
        for (int i = 0; i < realIndex; i++) {
            System.out.print(nums[i] + ",");
        }
        System.out.println("");
        System.out.println(realIndex + "");
    }

    /**
     * 有序数组保留连续2个数字
     *
     * @param nums
     */
    private static void removeSortedArray2(int[] nums) {
        if (nums.length == 0 || nums.length == 1) {
            return;
        }
        int value1 = nums[0];
        int value2 = nums[1];
        int count = 2;
        for (int i = count; i < nums.length; i++) {
            if (value1 != nums[i] && value2 != nums[i]) {//这个数是第1次出现
                value1 = nums[i];
                count++;
            } else if (value1 == nums[i] && value2 != nums[i] ||
                    value2 == nums[i] && value1 != nums[i]) {//这个数是第2次出现
                value2 = nums[i];
                count++;
            }
        }
        System.out.println(count + "");

    }

    /**
     * 有序数组保留连续3个数字
     *
     * @param nums
     */
    private static void removeSortedArray3(int[] nums) {
        if (nums.length == 0 || nums.length == 1) {
            return;
        }
        int value = nums[0];
        int onOff = 1;
        int count = 1;
        for (int i = 0; i < nums.length; i++) {
            if (value == nums[i] && (onOff <= 2)) {
                onOff++;
                count++;
            } else if (value != nums[i]) {
                value = nums[i];
                count++;
                onOff = 1;
            }
        }
        System.out.println(count + "");

    }

    /**
     * 75 Sort Colors
     * 给定一个有n个元素的数组，数组中元素的取值只有0,1,2三种可能，为这个数组排序
     */
    private static void SortColors(int[] nums) {
        int leftIndex = -1;
        int rightIndex = nums.length;
        for (int i = 0; i < rightIndex; ) {
            switch (nums[i]) {
                case 0: {
                    leftIndex++;
                    int temp = nums[i];
                    nums[i] = nums[leftIndex];
                    nums[leftIndex] = temp;
                    i++;
                }
                break;
                case 1:
                    i++;
                    break;
                case 2: {
                    rightIndex--;
                    int temp = nums[i];
                    nums[i] = nums[rightIndex];
                    nums[rightIndex] = temp;
                }
                break;
            }
        }


        for (int i = 0; i < nums.length; i++) {
            System.out.print(nums[i] + ",");
        }
    }

    private static void printArr(int[] arr) {
        for (int e : arr)
            System.out.print(e + " ");
        System.out.println();
    }

}
