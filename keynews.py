import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, CLIPModel
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

# 一带一路国家列表
BRI_COUNTRIES = {
    '亚洲': ['中国', '蒙古', '俄罗斯', '哈萨克斯坦', '吉尔吉斯斯坦', '塔吉克斯坦', '乌兹别克斯坦', 
            '土库曼斯坦', '越南', '老挝', '柬埔寨', '泰国', '马来西亚', '新加坡', '印度尼西亚', 
            '文莱', '菲律宾', '缅甸', '孟加拉国', '斯里兰卡', '马尔代夫', '印度', '巴基斯坦', 
            '尼泊尔', '不丹', '伊朗', '伊拉克', '土耳其', '叙利亚', '约旦', '黎巴嫩', '以色列', 
            '巴勒斯坦', '沙特阿拉伯', '也门', '阿曼', '阿联酋', '卡塔尔', '科威特', '巴林', 
            '希腊', '塞浦路斯', '埃及', '阿塞拜疆', '格鲁吉亚', '亚美尼亚', '摩尔多瓦', '乌克兰', 
            '白俄罗斯', '波兰', '立陶宛', '爱沙尼亚', '拉脱维亚', '捷克', '斯洛伐克', '匈牙利', 
            '斯洛文尼亚', '克罗地亚', '波黑', '黑山', '塞尔维亚', '阿尔巴尼亚', '罗马尼亚', 
            '保加利亚', '马其顿'],
    '非洲': ['埃及', '肯尼亚', '埃塞俄比亚', '坦桑尼亚', '南非', '尼日利亚', '加纳', '科特迪瓦', 
            '塞内加尔', '摩洛哥', '阿尔及利亚', '突尼斯', '利比亚', '苏丹', '南苏丹', '厄立特里亚', 
            '吉布提', '索马里', '乌干达', '卢旺达', '布隆迪', '刚果(金)', '刚果(布)', '加蓬', 
            '赤道几内亚', '圣多美和普林西比', '安哥拉', '赞比亚', '马拉维', '莫桑比克', '马达加斯加', 
            '科摩罗', '毛里求斯', '塞舌尔', '纳米比亚', '博茨瓦纳', '津巴布韦', '斯威士兰', '莱索托'],
    '欧洲': ['俄罗斯', '白俄罗斯', '乌克兰', '摩尔多瓦', '波兰', '立陶宛', '爱沙尼亚', '拉脱维亚', 
            '捷克', '斯洛伐克', '匈牙利', '斯洛文尼亚', '克罗地亚', '波黑', '黑山', '塞尔维亚', 
            '阿尔巴尼亚', '罗马尼亚', '保加利亚', '马其顿', '希腊', '塞浦路斯', '土耳其', '阿塞拜疆', 
            '格鲁吉亚', '亚美尼亚'],
    '大洋洲': ['澳大利亚', '新西兰', '巴布亚新几内亚', '斐济', '萨摩亚', '汤加', '瓦努阿图', 
              '密克罗尼西亚', '所罗门群岛', '基里巴斯', '图瓦卢', '瑙鲁', '马绍尔群岛', '帕劳'],
    '美洲': ['巴西', '阿根廷', '智利', '秘鲁', '哥伦比亚', '厄瓜多尔', '玻利维亚', '巴拉圭', 
            '乌拉圭', '委内瑞拉', '圭亚那', '苏里南', '巴拿马', '哥斯达黎加', '尼加拉瓜', 
            '洪都拉斯', '萨尔瓦多', '危地马拉', '伯利兹', '墨西哥', '古巴', '牙买加', '海地', 
            '多米尼加', '巴哈马', '巴巴多斯', '特立尼达和多巴哥', '圣基茨和尼维斯', '安提瓜和巴布达', 
            '多米尼克', '圣卢西亚', '圣文森特和格林纳丁斯', '格林纳达']
}

class MultiModalContentEncoder(nn.Module):
    def __init__(self, text_dim: int = 768, image_dim: int = 512, video_dim: int = 1024):
        super().__init__()
        # 文本编码器
        self.text_encoder = BertModel.from_pretrained('bert-base-chinese')
        # 图像编码器
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # 视频编码器
        self.video_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        
        # 跨模态注意力机制
        self.cross_modal_attention = nn.MultiheadAttention(embed_dim=text_dim, num_heads=8)
        
    def forward(self, text_input: torch.Tensor, image_input: torch.Tensor, 
                video_input: torch.Tensor) -> torch.Tensor:
        # 文本特征提取
        text_features = self.text_encoder(text_input).last_hidden_state.mean(dim=1)
        
        # 图像特征提取
        image_features = self.image_encoder.get_image_features(image_input)
        
        # 视频特征提取
        video_features = self.video_encoder(video_input)
        video_features = video_features.view(video_features.size(0), -1)
        
        # 跨模态注意力融合
        features = torch.stack([text_features, image_features, video_features], dim=1)
        attended_features, _ = self.cross_modal_attention(features, features, features)
        
        return attended_features.mean(dim=1)

class UserProfileAnalyzer:
    def __init__(self):
        self.user_profiles = defaultdict(dict)
        self.category_weights = {
            '政治': 0.2,
            '经济': 0.2,
            '科技': 0.2,
            '文化': 0.2,
            '体育': 0.2
        }
        
    def update_profile(self, user_id: str, interaction_data: Dict):
        """更新用户画像"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'interests': defaultdict(float),
                'reading_time': defaultdict(float),
                'interaction_frequency': defaultdict(int),
                'preferred_categories': set(),
                'reading_habits': {
                    'morning': 0,
                    'afternoon': 0,
                    'evening': 0
                },
                'country_preferences': {
                    'home_country': None,
                    'interested_countries': set(),
                    'country_interaction_freq': defaultdict(int),
                    'region_preferences': defaultdict(float)
                }
            }
        
        profile = self.user_profiles[user_id]
        
        # 更新国家偏好
        if 'home_country' in interaction_data:
            profile['country_preferences']['home_country'] = interaction_data['home_country']
        
        if 'interested_countries' in interaction_data:
            profile['country_preferences']['interested_countries'].update(
                interaction_data['interested_countries']
            )
        
        if 'country_interactions' in interaction_data:
            for country, freq in interaction_data['country_interactions'].items():
                profile['country_preferences']['country_interaction_freq'][country] += freq
        
        # 更新地区偏好
        if 'region_interactions' in interaction_data:
            for region, weight in interaction_data['region_interactions'].items():
                profile['country_preferences']['region_preferences'][region] = (
                    profile['country_preferences']['region_preferences'][region] * 0.7 + 
                    weight * 0.3
                )
        
        # 原有的更新逻辑
        for category, weight in interaction_data.get('categories', {}).items():
            profile['interests'][category] = (profile['interests'][category] * 0.7 + 
                                            weight * 0.3)
        
        for time_slot, duration in interaction_data.get('reading_time', {}).items():
            profile['reading_time'][time_slot] = (profile['reading_time'][time_slot] * 0.7 + 
                                                duration * 0.3)
        
        for category in interaction_data.get('interacted_categories', []):
            profile['interaction_frequency'][category] += 1
        
        profile['preferred_categories'].update(
            interaction_data.get('preferred_categories', [])
        )
        
        for time_slot in interaction_data.get('reading_habits', []):
            profile['reading_habits'][time_slot] += 1

class EnhancedUserModel(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        self.short_term_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.long_term_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8),
            num_layers=3
        )
        
        # 用户画像特征提取
        self.profile_encoder = nn.Sequential(
            nn.Linear(5, 64),  # 5个主要新闻类别
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        # 国家偏好特征提取
        self.country_encoder = nn.Sequential(
            nn.Linear(len(BRI_COUNTRIES), 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        
        # 时间衰减权重
        self.time_decay = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, user_history: torch.Tensor, 
                time_stamps: torch.Tensor,
                user_profile: Dict) -> torch.Tensor:
        # 短期兴趣
        short_term, _ = self.short_term_lstm(user_history)
        short_term = short_term[:, -1, :]
        
        # 长期兴趣
        long_term = self.long_term_transformer(user_history)
        long_term = long_term.mean(dim=1)
        
        # 用户画像特征
        profile_features = self.profile_encoder(
            torch.tensor([user_profile['interests'][cat] 
                         for cat in ['政治', '经济', '科技', '文化', '体育']])
        )
        
        # 国家偏好特征
        country_features = self.country_encoder(
            torch.tensor([
                user_profile['country_preferences']['region_preferences'].get(region, 0.0)
                for region in BRI_COUNTRIES.keys()
            ])
        )
        
        # 时间衰减
        time_weights = torch.exp(-self.time_decay * time_stamps)
        time_weights = time_weights / time_weights.sum(dim=1, keepdim=True)
        
        # 动态融合
        user_embedding = torch.sum(time_weights.unsqueeze(-1) * user_history, dim=1)
        
        # 融合用户画像和国家偏好
        user_embedding = user_embedding + profile_features + country_features
        
        return user_embedding

class PersonalizedNewsRecommender(nn.Module):
    def __init__(self, content_dim: int = 768, user_dim: int = 512):
        super().__init__()
        self.content_encoder = MultiModalContentEncoder()
        self.user_model = EnhancedUserModel()
        self.profile_analyzer = UserProfileAnalyzer()
        
        # 知识图谱增强
        self.gnn = nn.Sequential(
            nn.Linear(content_dim, 256),
            nn.ReLU(),
            nn.Linear(256, content_dim)
        )
        
        # 个性化推荐预测
        self.predictor = nn.Sequential(
            nn.Linear(content_dim + user_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, content_inputs: Dict[str, torch.Tensor],
                user_history: torch.Tensor,
                time_stamps: torch.Tensor,
                user_id: str) -> torch.Tensor:
        # 内容编码
        content_features = self.content_encoder(
            content_inputs['text'],
            content_inputs['image'],
            content_inputs['video']
        )
        
        # 知识图谱增强
        enhanced_content = self.gnn(content_features)
        
        # 获取用户画像
        user_profile = self.profile_analyzer.user_profiles[user_id]
        
        # 用户建模（包含画像信息）
        user_features = self.user_model(user_history, time_stamps, user_profile)
        
        # 推荐预测
        combined_features = torch.cat([enhanced_content, user_features], dim=1)
        prediction = self.predictor(combined_features)
        
        return prediction

class NewsRecommendationSystem:
    def __init__(self):
        self.model = PersonalizedNewsRecommender()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
    def update_user_profile(self, user_id: str, interaction_data: Dict):
        """更新用户画像"""
        self.model.profile_analyzer.update_profile(user_id, interaction_data)
        
    def train(self, train_data: Dict[str, torch.Tensor], epochs: int = 10):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            # 前向传播
            predictions = self.model(
                train_data['content'],
                train_data['user_history'],
                train_data['time_stamps'],
                train_data['user_id']
            )
            
            # 计算损失
            loss = F.binary_cross_entropy(predictions, train_data['labels'])
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
            
    def recommend(self, content_inputs: Dict[str, torch.Tensor],
                 user_history: torch.Tensor,
                 time_stamps: torch.Tensor,
                 user_id: str,
                 top_k: int = 10) -> List[int]:
        self.model.eval()
        with torch.no_grad():
            scores = self.model(content_inputs, user_history, time_stamps, user_id)
            top_indices = torch.topk(scores, k=top_k)[1].tolist()
        return top_indices
