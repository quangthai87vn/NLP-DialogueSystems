# Demo: GlobalWoZ Vietnamese (F2E)

- Source: `data/globalwoz/F2E_vi.json`
- Dialogue: `MUL0003.json`

## Goal (belief target)
```json
{
  "attraction": {},
  "hospital": {},
  "hotel": {
    "book": {
      "day": "sunday",
      "invalid": false,
      "people": "6",
      "stay": "4"
    },
    "fail_book": {},
    "fail_info": {},
    "info": {
      "internet": "yes",
      "parking": "yes",
      "pricerange": "cheap",
      "type": "guesthouse"
    }
  },
  "police": {},
  "restaurant": {
    "book": {
      "day": "sunday",
      "invalid": false,
      "people": "6",
      "time": "18:45"
    },
    "fail_book": {},
    "fail_info": {},
    "info": {
      "area": "centre",
      "food": "italian",
      "pricerange": "cheap"
    }
  },
  "taxi": {},
  "train": {}
}
```

## Turns (User/System)

**Turn 0 – User:** tôi đang tìm một nơi để ở. nó phải là guesthouse và có wifi miễn phí .
**Turn 1 – System:** có 23 hotels đáp ứng nhu cầu của bạn . bạn có muốn thu hẹp tìm kiếm của mình theo khu vực và / hoặc phạm vi giá không ?
**Turn 2 – User:** tôi muốn nó là cheap và có chỗ đậu xe miễn phí .
**Turn 3 – System:** có chín guesthouse hotels trong các khu vực various . bạn đang hy vọng ở khu vực nào của thị trấn ?
**Turn 4 – User:** không có gì đặc biệt . tôi chỉ cần đặt phòng cho 6 người với tổng số 4 đêm bắt đầu từ sunday . tôi cũng muốn số tham chiếu , xin vui lòng .
**Turn 5 – System:** chắc chắn rồi ! chúng tôi sẽ đặt bạn ở hotels với 4 sao . đặt phòng sẽ dành cho 6 người cho 4 đêm . bạn muốn đặt bao nhiêu phòng ?
**Turn 6 – User:** tốt , đối với 6 , tôi cho rằng tôi sẽ cần ba phòng , cho 4 đêm , bắt đầu từ sunday .
**Turn 7 – System:** bạn đã đặt phòng tại alexander bed and breakfast , 517a coldham lane , cho 6 cho 4 đêm trong sunday . số tham chiếu của bạn là jxvkz7kv .
**Turn 8 – User:** Cảm ơn ! tôi cũng đang tìm một nhà hàng . lý tưởng là một địa điểm italian có cùng mức giá trong centre .
**Turn 9 – System:** tôi đã tìm thấy nhà hàng ask , bạn có muốn đặt chỗ đó không ?
**Turn 10 – User:** là tên của nhà hàng ask nhà hàng ?
**Turn 11 – System:** có , nhà hàng được gọi là ask . bạn có muốn đặt bàn ở đó không ?
**Turn 12 – User:** đúng . đặt phòng cho 6 vào sunday tại 18:45 . Tôi có thể lấy số tham chiếu được không ?
**Turn 13 – System:** tuyệt vời ! đặt phòng của bạn cho 6 tại nhà hàng ask vào sunday lúc 18:45 đã thành công ! số tham chiếu của bạn là y2y8qyby .
**Turn 14 – User:** tôi đã sẵn sàng . Cảm ơn .
**Turn 15 – System:** cảm ơn bạn đã sử dụng tổng đài hỗ trợ towneinfo TPHCM !